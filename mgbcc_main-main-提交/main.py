import argparse
import os.path as path

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchsummary import summary

from granular.base import MVGBList
from granular.granular_loss import MultiviewGCLoss
from model.autoencoder import MultiviewAutoEncoder, Normalize
from model.loss import ContrastiveLoss
from utils.common import load_json, init_torch
from utils.dataset import MultiviewDataset
from utils.score import cluster_metric
from itertools import product
import numpy as np

import warnings
warnings.filterwarnings("ignore")
from leading_tree import GB_LeadingTree
from granular.granular_loss import HierarchicalTreeLoss

def run_on_setting(args, **kwargs):
    # 读取全局参数
    global_params = load_json(args.global_config)
    # 设置数据集相关设置的路径，默认与全局配置在同一文件夹下
    if global_params["config"] is None:
        config_path = path.dirname(args.global_config)
        global_params["config"] = path.join(config_path, args.dataset + ".json")
    # 随机数种子
    init_torch(seed=global_params["seed"])
    # 读取数据集特定的参数
    ds_config = load_json(global_params["config"])

    # 优先级：kwargs > ds_config > global_params
    for key in global_params:
        if key not in ds_config:
            ds_config[key] = global_params[key]
    for key in kwargs:
        ds_config[key] = kwargs[key]
    # 读取数据集
    src = ds_config["src"]
    ds_path = path.join(src, ds_config["parent"], args.dataset + ".mat")
    mv_dataset = MultiviewDataset(ds_path, ds_config["device"], views=ds_config["select_views"],
                                  normalize=ds_config["normalize"])
    # 构建数据加载器
    batch_size = ds_config["batch_size"]
    if batch_size == -1:
        batch_size = len(mv_dataset)
    # 如果是先基于cpu加载数据，再放到gpu上，则num_workers可以设置大于0
    dataloader = DataLoader(mv_dataset, batch_size, shuffle=True, num_workers=0)
    # 构建模型
    mv_aes = MultiviewAutoEncoder(mv_dataset.view_dims, ds_config["latent_dim"],
                                  ds_config["autoencoder"]["mid_archs"], ds_config["use_linear_projection"])
    # 在编码层后，加一层标准化层
    for v in range(mv_dataset.num_view):
        mv_aes[v].encoder.middle_layers.append(Normalize())
    mv_aes.to(ds_config["device"])
    summary(mv_aes[0], (mv_dataset.view_dims[0],))
    # 优化器
    optimizer = torch.optim.Adam(mv_aes.parameters(),
                                 lr=ds_config["learning_rate"],
                                 weight_decay=ds_config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=ds_config["epochs"], eta_min=0.)

    weights = ds_config["loss_weights"]
    kmeans = KMeans(n_clusters=mv_dataset.num_class, n_init="auto", random_state=ds_config["seed"])
    criterion_rec = nn.MSELoss()
    criterion_gra = MultiviewGCLoss()
    criterion_ins = ContrastiveLoss()
    criterion_tree = HierarchicalTreeLoss(temperature=1.0)
    tree_loss_weight = ds_config.get("tree_loss_weight", 0.1)

    result = {
        "epoch": [],
        "loss_con": [],
        "loss_rec": [],
        "loss_tree": [],
        "loss_total": [],
        "ACC": [],
        "NMI": [],
        "PUR": [],
        "sh": [],
        "ch": [],
        "db": []
    }

    for epoch in range(ds_config["epochs"]):
        loss_con_avg = 0
        loss_rec_avg = 0
        loss_tree_avg = 0
        loss_total_avg = 0
        mv_aes.train()
        for bid, (x, y) in enumerate(dataloader):
            hs, x_rs = mv_aes(x)

            # 1. 重建损失
            loss_rec = torch.tensor(0., device=ds_config["device"])
            for v in range(mv_dataset.num_view):
                loss_rec += criterion_rec(x[v], x_rs[v])

            loss_tree = torch.tensor(0., device=ds_config["device"])  # 初始化当前 batch 的树损失

            if ds_config["p"] > 1:
                # 1. 构建多视图粒球对象
                mv_gblist = MVGBList(hs, y, ds_config["p"])
                # 2. 构建引领树
                loss_tree = torch.tensor(0., device=ds_config["device"])
                # 热身机制。前 20 轮只做原版对比，20轮后开启树的指导
                warmup_epochs = ds_config.get("start_dual_prediction", 20)

                if epoch >= warmup_epochs:
                    for v in range(mv_dataset.num_view):
                        centers_np, rs_np, counts_np = mv_gblist[v].get_stats_for_tree()
                        lt = GB_LeadingTree(centers_np, rs_np, counts_np, lt_num=mv_dataset.num_class)
                        lt.fit()
                        tree_mask_np = lt.get_tree_mask()

                        # 将 Numpy 的层级数组转为 Tensor，并挂载到当前的粒球视图对象上
                        mv_gblist[v].tree_layers = torch.from_numpy(lt.layer).to(ds_config["device"]).float()
                        # 计算视图内的偏序树损失
                        loss_tree += criterion_tree(mv_gblist[v], tree_mask_np)
                    loss_tree = loss_tree / mv_dataset.num_view
                # 3. 后计算跨视图对比损失
                loss_con = criterion_gra(mv_gblist)

            else:
                loss_con, _, _ = criterion_ins(hs)
                loss_tree = torch.tensor(0., device=ds_config["device"])
            # 将树损失融入总 Loss 中进行反向传播
            loss = loss_con * weights[0] + loss_rec * weights[1] + loss_tree * tree_loss_weight
            loss_con_avg += loss_con.item()
            loss_rec_avg += loss_rec.item()
            loss_tree_avg += loss_tree.item() if isinstance(loss_tree, torch.Tensor) else loss_tree
            loss_total_avg += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_con_avg /= len(dataloader)
        loss_rec_avg /= len(dataloader)
        loss_tree_avg /= len(dataloader)
        loss_total_avg /= len(dataloader)

        scheduler.step()
        mv_aes.eval()
        with torch.no_grad():
            hs, _ = mv_aes(mv_dataset.data)
            # 拼接不同视图的特征
            hs = torch.stack(hs, dim=0).mean(0).detach().cpu().numpy()
            # k_means
            y_pred = kmeans.fit_predict(hs)
            y_true = mv_dataset.labels.cpu().numpy()
            acc, nmi, pur = cluster_metric(y_true, y_pred)
            acc, nmi, pur = round(acc, 4) * 100, round(nmi, 4) * 100, round(pur, 4) * 100
            sh = silhouette_score(hs, y_pred, metric='euclidean')
            ch = calinski_harabasz_score(hs, y_pred)
            db = davies_bouldin_score(hs, y_pred)

            print(f"epoch {epoch + 1}, loss_total: {round(loss_total_avg, 4):.4f}, "
                  f"loss_con: {round(loss_con_avg, 4):.4f}, loss_rec: {round(loss_rec_avg, 4):.4f}, "
                  f"loss_tree: {round(loss_tree_avg, 4):.4f}, "
                  f"acc {acc:.2f}%, nmi {nmi:.2f}%, pur {pur:.2f}%")

            result["epoch"].append(epoch)
            result["loss_con"].append(loss_con_avg)
            result["loss_rec"].append(loss_rec_avg)
            result["loss_tree"].append(loss_tree_avg)
            result["loss_total"].append(loss_total_avg)
            result["ACC"].append(acc)
            result["NMI"].append(nmi)
            result["PUR"].append(pur)
            result["sh"].append(sh)
            result["ch"].append(ch)
            result["db"].append(db)
    return result

def run():
    # 读取命令行参数
    parser = argparse.ArgumentParser(description="Command Line Params")
    parser.add_argument("--dataset", type=str, default="Caltech101-20", help="Dataset used for Training.")
    parser.add_argument("--global_config", type=str, default="./config/granular_config/global.json", help="The path of global config files.")
    args = parser.parse_args()
    # run_on_setting(args)
    # 参数搜索
    learning_rate = [1e-4]
    p_values = [3]
    latent_dim = [128]
    batch_size = [64, 128, 256, 512, -1]  # -1 代表 Full-batch
    normalize = [True]
    tree_loss_weights = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    # 生成参数组合迭代器
    parameter_iter = product(latent_dim, p_values, learning_rate, normalize, batch_size, tree_loss_weights)
    dataset = args.dataset

    if not path.exists("./result"):
        os.makedirs("./result")

    save_path = "./result/" + dataset + "_leading_tree.dat"
    f = open(save_path, "w")
    print(f"=== 开始在 {dataset} 数据集上运行【基于引领树指导的粒球聚类】===")
    # 开始网格搜索遍历
    for latent_dim, p, learning_rate, normalize, batch_size, tree_loss_weight in parameter_iter:

        # 打印当前正在运行的参数组合
        setting_info = (f"latent_dim={latent_dim}, p={p}, lr={learning_rate}, "
                        f"norm={normalize}, bs={batch_size}, tree_w={tree_loss_weight}")
        print(f"\n---> Running: {setting_info}")

        try:
            result = run_on_setting(
                args,
                batch_size=batch_size,
                latent_dim=latent_dim,
                p=p,
                learning_rate=learning_rate,
                normalize=normalize,
                dataset=dataset,
                tree_loss_weight=tree_loss_weight
            )

            # 记录过程中的最佳实验结果和最终实验结果
            best_epoch = np.argmax(result["ACC"])
            best_result = {
                "epoch": best_epoch,
                "ACC": result["ACC"][best_epoch],
                "NMI": result["NMI"][best_epoch],
                "PUR": result["PUR"][best_epoch],
                "sh": result["sh"][best_epoch],
                "ch": result["ch"][best_epoch],
                "db": result["db"][best_epoch]
            }
            final_result = {
                "epoch": result["epoch"][-1],
                "ACC": result["ACC"][-1],
                "NMI": result["NMI"][-1],
                "PUR": result["PUR"][-1],
                "sh": result["sh"][-1],
                "ch": result["ch"][-1],
                "db": result["db"][-1]
            }

            # 控制台输出结果
            print(f"✅ Success | {setting_info}")
            print(f"   Best Result : {best_result}")
            print(f"   Final Result: {final_result}")

            # 将结果写入文件
            f.write(f"[{setting_info}] | best_result={best_result} | final_result={final_result}\n")

        except Exception as e:
            # 捕获异常，打印错误堆栈，但不中断后续的参数搜索
            print(f"❌ Failed | {setting_info}")
            print(f"   Error Message: {str(e)}\n")
            f.write(f"[{setting_info}] | Failed: {str(e)}\n")

        # 每跑完一组参数强制刷新缓存，防止跑了一半程序崩溃导致记录丢失
        f.flush()

    f.close()
    print("\n=== 所有参数搜索完成 ===")


if __name__ == "__main__":
    run()



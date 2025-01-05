import numpy as np
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from BPP import Bin, Item, Packer, Painter  

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义可用的箱子尺寸
AVAILABLE_BOX_SIZES = [
    (35, 23, 13),
    (37, 26, 13),
    (38, 26, 13),
    (40, 28, 16),
    (42, 30, 18),
    (42, 30, 40),
    (52, 40, 17),
    (54, 45, 36),
]

def create_items_from_order(order):
    """将订单中的商品数据转换为 Item 对象"""
    items = []
    for sku_code, product in order['products'].items():
        for _ in range(product['qty']):
            items.append(
                Item(
                    partno=sku_code,
                    name=sku_code,
                    typeof='cube',
                    WHD=(product['length'], product['width'], product['height']),
                    weight=0,  # 忽略重量
                    level=1,  # 优先级默认为1
                    loadbear=0,  # 忽略承重
                    updown=True,  # 允许旋转
                    color="#FFFFFF"  # 默认颜色
                )
            )
    return items

def pack_order_with_multiple_boxes(order):
    """为单个订单进行包装，允许使用多个箱子"""
    items = create_items_from_order(order)

    used_bins = []  # 用于存储所使用的箱子
    remaining_items = items.copy()  # 待装箱的商品列表

    box_instance_counter = 0  # 用于记录箱子实例的唯一编号

    while remaining_items:
        # 按箱子体积从小到大尝试找到合适的箱子
        for box_size in sorted(AVAILABLE_BOX_SIZES, key=lambda x: x[0] * x[1] * x[2]):
            box_instance_counter += 1
            bin = Bin(partno=f"Bin-{box_instance_counter}", WHD=box_size, max_weight=1000)

            packer = Packer()
            packer.addBin(bin)

            # 添加剩余商品
            for item in remaining_items:
                packer.addItem(item)

            # 尝试装箱
            packer.pack()

            # 如果有装进去的物品，记录当前箱子并移除已装物品
            if bin.items:
                used_bins.append({
                    "box_size": box_size,
                    "bin_id": f"箱子-{box_instance_counter}",
                    "packed_items": [
                        {
                            "sku_code": item.partno,
                            "position": item.position,
                            "rotation": item.rotation_type,
                            "width": item.getDimension()[0],  # 宽度
                            "height": item.getDimension()[1],  # 高度
                            "depth": item.getDimension()[2],  # 深度
                        }
                        for item in bin.items
                    ],
                    "utilization": sum([item.getVolume() for item in bin.items]) / (box_size[0] * box_size[1] * box_size[2]) * 100
                })
                remaining_items = bin.unfitted_items  # 更新未能装箱的商品
                break

    # 可视化每个箱子
    for bin_info in used_bins:
        box_size = bin_info['box_size']
        box = Bin(partno=bin_info['bin_id'], WHD=box_size, max_weight=1000)

        # 创建 Item 对象，并设置其 position 和 rotation
        for item_info in bin_info['packed_items']:
            item = Item(
                partno=item_info['sku_code'],
                name=item_info['sku_code'],  # 使用 SKU 代码作为名称
                typeof='cube',  # 假设都是立方体
                WHD=(item_info['width'], item_info['height'], item_info['depth']),
                weight=0,  # 忽略重量
                level=1,  # 默认优先级
                loadbear=0,  # 忽略承重
                updown=True,  # 允许旋转
                color="#FFFFFF"  # 默认颜色
            )
            item.position = item_info['position']  # 设置位置
            item.rotation_type = item_info['rotation']  # 设置旋转类型
            box.items.append(item)

        painter = Painter(box)  # 创建 Painter 对象
        title_text = f"订单: {order['sta_code']} - 尺寸: {bin_info['box_size'][0]}x{bin_info['box_size'][1]}x{bin_info['box_size'][2]}"  # 显示订单号和箱子尺寸

        # 绘制箱子和物品，考虑旋转类型
        painter.plotBoxAndItems(title=title_text, alpha=0.5, write_num=True, fontsize=10)
        plt.show()  # 显示每个箱子的可视化

    return {
        "sta_code": order['sta_code'],
        "used_bins": used_bins,
        "total_utilization": sum(bin['utilization'] for bin in used_bins) / len(used_bins)
    }

# 示例订单数据
orders = [
    {
        "sta_code": "BSIN2309027896",
        "products": {
            "DA2123-100": {"length": 49, "width": 25, "height": 7, "qty": 1},
            "DD9535-007": {"length": 14.1, "width": 9.9, "height": 6.4, "qty": 3},

        },
    }
]

# 对每个订单进行装箱并输出结果
for order in orders:
    result = pack_order_with_multiple_boxes(order)
    print(f"订单 {result['sta_code']} 的包装结果：")
    print(f"  总体利用率：{result['total_utilization']:.2f}%")
    for bin_info in result['used_bins']:
        print(f"  箱子 {bin_info['bin_id']} (尺寸 {bin_info['box_size']}):")
        print(f"    利用率：{bin_info['utilization']:.2f}%")
        for item in bin_info['packed_items']:
            print(f"    商品 {item['sku_code']} -> 位置 {item['position']}, 旋转类型 {item['rotation']}")
    print("\n")
import pandas as pd
from BPP import Bin, Item, Packer

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
    for sku_code, product in order.items():
        for _ in range(product['qty']):
            items.append(
                Item(
                    partno=sku_code,
                    name=sku_code,
                    typeof='cube',
                    WHD=(product['length'], product['width'], product['height']),
                    weight=0,
                    level=1,
                    loadbear=0,
                    updown=True,
                    color="#FFFFFF"
                )
            )
    return items


def pack_order_with_multiple_boxes(sta_code, order):
    """为单个订单进行包装，允许使用多个箱子"""
    items = create_items_from_order(order)
    used_bins = []
    remaining_items = items.copy()
    box_instance_counter = 0

    total_item_volume = sum(item.getVolume() for item in items)  # 计算所有物品的总体积
    total_box_volume = 0  # 初始化总箱子体积

    while remaining_items:
        all_items_fit_in_bins = False
        for box_size in sorted(AVAILABLE_BOX_SIZES, key=lambda x: x[0] * x[1] * x[2]):
            box_volume = box_size[0] * box_size[1] * box_size[2]
            box_instance_counter += 1
            bin = Bin(partno=f"Bin-{box_instance_counter}", WHD=box_size, max_weight=1000)
            packer = Packer()
            packer.addBin(bin)
            for item in remaining_items:
                packer.addItem(item)
            packer.pack()
            if bin.items:
                used_bins.append({
                    "sta_code": sta_code,
                    "box_id": f"{box_size[0]}x{box_size[1]}x{box_size[2]}",
                    "box_size": box_size,
                    "packed_items": [
                        {
                            "sku_code": item.partno,
                            "position": item.position,
                            "rotation": item.rotation_type,
                            "is_rotated": item.rotation_type is not None,
                            "rotation_type": item.rotation_type
                        }
                        for item in bin.items
                    ],
                    "utilization": sum([item.getVolume() for item in bin.items]) / box_volume * 100
                })
                total_box_volume += box_volume  # 累加箱子的总体积
                remaining_items = bin.unfitted_items
                all_items_fit_in_bins = True
                break
        if not all_items_fit_in_bins:
            too_large_items = [item for item in remaining_items if item.getVolume() > box_volume]
            if too_large_items:
                return [], total_item_volume, total_box_volume  # 返回空列表和体积
            return [], total_item_volume, total_box_volume  # 返回空列表和体积

    return used_bins, total_item_volume, total_box_volume  # 返回箱子信息和体积


def process_task3_csv(input_file, output_file):
    # 读取 CSV 数据
    data = pd.read_csv(input_file, encoding='utf-8')
    data.rename(columns={"长(CM)": "length", "宽(CM)": "width", "高(CM)": "height"}, inplace=True)

    # 按 sta_code 分组
    results = []

    overall_total_item_volume = 0  # 所有物品总容量
    overall_total_box_volume = 0  # 所有箱子总容量

    for sta_code, group in data.groupby('sta_code'):
        order = {}
        for _, row in group.iterrows():
            sku_code = row['sku_code']
            if sku_code not in order:
                order[sku_code] = {
                    'length': row['length'],
                    'width': row['width'],
                    'height': row['height'],
                    'qty': row['qty']
                }
            else:
                order[sku_code]['qty'] += row['qty']

        packed_bins, total_item_volume, total_box_volume = pack_order_with_multiple_boxes(sta_code, order)

        # 更新整体总容量
        overall_total_item_volume += total_item_volume
        overall_total_box_volume += total_box_volume

        if isinstance(packed_bins, dict) and 'error' in packed_bins:
            results.append({
                "sta_code": sta_code,
                "box_id": "Error",
                "sku_code": ";".join(packed_bins.get('too_large_items', [])),
                "utilization": "N/A",
                "error": packed_bins['error'],
                "is_rotated": "N/A",
                "rotation_type": "N/A"
            })
        else:
            for bin_info in packed_bins:
                for item in bin_info['packed_items']:
                    results.append({
                        "sta_code": bin_info['sta_code'],
                        "box_id": bin_info['box_id'],
                        "sku_code": item['sku_code'],
                        "position": item['position'],
                        "rotation": item['rotation'],
                        "utilization": bin_info['utilization'],
                        "is_rotated": item['is_rotated'],
                        "rotation_type": item['rotation_type']
                    })

    # 输出整体容积率
    if overall_total_box_volume > 0:
        overall_volume_utilization = (overall_total_item_volume / overall_total_box_volume) * 100
        print(f"Overall Total Volume Utilization: {overall_volume_utilization:.2f}%")

    # 保存到 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"结果已保存到 {output_file}")


# 执行任务
input_csv = "task3.csv"
output_csv = "task3_output.csv"
process_task3_csv(input_csv, output_csv)
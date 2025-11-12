#!/usr/bin/env python3
"""
点云颜色调试工具
用于分析和修复OneFormer3D中的颜色问题
"""

import numpy as np
import open3d as o3d
import os
import argparse
from pathlib import Path


def analyze_point_cloud_colors(pc_path):
    """分析点云文件的颜色信息"""
    print(f"分析点云文件: {pc_path}")
    
    if not os.path.exists(pc_path):
        print(f"文件不存在: {pc_path}")
        return None
    
    # 读取点云
    pc = o3d.io.read_point_cloud(pc_path)
    if not pc.has_colors():
        print("点云没有颜色信息")
        return None
    
    # 获取颜色数据
    colors = np.asarray(pc.colors)
    
    print(f"颜色形状: {colors.shape}")
    print(f"颜色数据类型: {colors.dtype}")
    print(f"颜色范围: {colors.min():.6f} - {colors.max():.6f}")
    print(f"颜色均值: {colors.mean(axis=0)}")
    print(f"颜色标准差: {colors.std(axis=0)}")
    
    # 检查是否有异常值
    zero_colors = np.sum(colors == 0, axis=1)
    one_colors = np.sum(colors == 1, axis=1)
    print(f"全零颜色点数: {np.sum(zero_colors == 3)}")
    print(f"全一颜色点数: {np.sum(one_colors == 3)}")
    
    return colors


def fix_color_range(pc_path, output_path, target_range='0-1'):
    """修复点云颜色范围"""
    print(f"修复颜色范围: {pc_path} -> {output_path}")
    
    pc = o3d.io.read_point_cloud(pc_path)
    if not pc.has_colors():
        print("点云没有颜色信息")
        return
    
    colors = np.asarray(pc.colors)
    original_range = f"{colors.min():.3f}-{colors.max():.3f}"
    print(f"原始颜色范围: {original_range}")
    
    # 根据目标范围进行修复
    if target_range == '0-1':
        if colors.max() > 1.0:
            print("检测到颜色值大于1.0，进行255归一化")
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
    elif target_range == '0-255':
        if colors.max() <= 1.0:
            print("检测到颜色值在0-1范围内，转换为0-255")
            colors = colors * 255.0
        colors = np.clip(colors, 0.0, 255.0)
    
    # 更新点云颜色
    pc.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存修复后的点云
    o3d.io.write_point_cloud(output_path, pc)
    print(f"修复后颜色范围: {colors.min():.3f}-{colors.max():.3f}")
    print(f"已保存到: {output_path}")


def compare_point_clouds(pc1_path, pc2_path):
    """比较两个点云的颜色差异"""
    print(f"比较点云颜色: {pc1_path} vs {pc2_path}")
    
    pc1 = o3d.io.read_point_cloud(pc1_path)
    pc2 = o3d.io.read_point_cloud(pc2_path)
    
    if not pc1.has_colors() or not pc2.has_colors():
        print("至少一个点云没有颜色信息")
        return
    
    colors1 = np.asarray(pc1.colors)
    colors2 = np.asarray(pc2.colors)
    
    print(f"点云1颜色范围: {colors1.min():.3f}-{colors1.max():.3f}")
    print(f"点云2颜色范围: {colors2.min():.3f}-{colors2.max():.3f}")
    
    # 计算颜色差异
    if colors1.shape == colors2.shape:
        diff = np.abs(colors1 - colors2)
        print(f"平均颜色差异: {diff.mean():.6f}")
        print(f"最大颜色差异: {diff.max():.6f}")
        print(f"颜色差异标准差: {diff.std():.6f}")
    else:
        print("两个点云的点数不同，无法直接比较")


def create_color_visualization(pc_path, output_dir):
    """创建颜色可视化"""
    print(f"创建颜色可视化: {pc_path}")
    
    pc = o3d.io.read_point_cloud(pc_path)
    if not pc.has_colors():
        print("点云没有颜色信息")
        return
    
    colors = np.asarray(pc.colors)
    points = np.asarray(pc.points)
    
    # 创建不同颜色通道的可视化
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 红色通道
    red_pc = o3d.geometry.PointCloud()
    red_pc.points = o3d.utility.Vector3dVector(points)
    red_colors = np.zeros_like(colors)
    red_colors[:, 0] = colors[:, 0]  # 只保留红色通道
    red_pc.colors = o3d.utility.Vector3dVector(red_colors)
    o3d.io.write_point_cloud(str(output_dir / "red_channel.ply"), red_pc)
    
    # 绿色通道
    green_pc = o3d.geometry.PointCloud()
    green_pc.points = o3d.utility.Vector3dVector(points)
    green_colors = np.zeros_like(colors)
    green_colors[:, 1] = colors[:, 1]  # 只保留绿色通道
    green_pc.colors = o3d.utility.Vector3dVector(green_colors)
    o3d.io.write_point_cloud(str(output_dir / "green_channel.ply"), green_pc)
    
    # 蓝色通道
    blue_pc = o3d.geometry.PointCloud()
    blue_pc.points = o3d.utility.Vector3dVector(points)
    blue_colors = np.zeros_like(colors)
    blue_colors[:, 2] = colors[:, 2]  # 只保留蓝色通道
    blue_pc.colors = o3d.utility.Vector3dVector(blue_colors)
    o3d.io.write_point_cloud(str(output_dir / "blue_channel.ply"), blue_pc)
    
    print(f"颜色通道可视化已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="点云颜色调试工具")
    parser.add_argument("command", choices=["analyze", "fix", "compare", "visualize"], 
                       help="要执行的命令")
    parser.add_argument("input", help="输入点云文件路径")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--target-range", choices=["0-1", "0-255"], default="0-1",
                       help="目标颜色范围")
    parser.add_argument("--pc2", help="第二个点云文件路径（用于比较）")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyze_point_cloud_colors(args.input)
    
    elif args.command == "fix":
        if not args.output:
            print("修复命令需要指定输出路径")
            return
        fix_color_range(args.input, args.output, args.target_range)
    
    elif args.command == "compare":
        if not args.pc2:
            print("比较命令需要指定第二个点云文件")
            return
        compare_point_clouds(args.input, args.pc2)
    
    elif args.command == "visualize":
        output_dir = args.output or "color_visualization"
        create_color_visualization(args.input, output_dir)


if __name__ == "__main__":
    main()

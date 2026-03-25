import os
import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # 配色完全参考上传的图片
    # 第一个输入层：深蓝
    # 两个隐藏层：青蓝色
    # 最后一个全连接层/输出：浅青蓝色
    node_colors = ['#042663', '#0dd3d6', '#0dd3d6', '#0dd3d6']
    edge_colors = ['#1d4b8f', '#1999b8', '#1999b8']

    # 绘制连线
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], 
                                  c=edge_colors[n], alpha=0.7, lw=1.2)
                ax.add_artist(line)
                
    # 绘制节点
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.5,
                                color=node_colors[n], zorder=4)
            ax.add_artist(circle)

fig, ax = plt.subplots(figsize=(7, 5))
ax.axis('off')

# 代码中的模型结构是：
# 1. 输入层: nn.LSTM(input_size=1)，表示每个时间步只输入1个变量（单变量吞吐量），所以视觉上用 1 个节点表示。
# 2. LSTM 第一层 (num_layers=2) (视觉上用 6 个节点表示隐藏状态)
# 3. LSTM 第二层 (视觉上用 6 个节点表示提取的隐藏特征)
# 4. 全连接输出层 (FC): nn.Linear(hidden, 1)，预测单个吞吐量值，用 1 个节点表示。
# 调整网络层节点数: 1(Input) -> 10(LSTM Layer1) -> 10(LSTM Layer2) -> 1(FC Output)
draw_neural_net(ax, 0.1, 0.9, 0.15, 0.85, [1, 10, 10, 1])

plt.text(0.1, 0.95, 'LSTM', fontsize=18, fontweight='bold', ha='center', color='black')

# 底部文字说明，体现代码中的两层 LSTM 结构
plt.text(0.1, 0.0, 'Input\n(Hist Sequence)', fontsize=11, ha='center')
plt.text(0.36, 0.0, 'LSTM Layer 1\n(Hidden=64)', fontsize=11, ha='center')
plt.text(0.63, 0.0, 'LSTM Layer 2\n(Hidden=64)', fontsize=11, ha='center')
plt.text(0.9, 0.0, 'Linear(FC)\n(Prediction)', fontsize=11, ha='center')

os.makedirs('assets/images', exist_ok=True)
plt.savefig('assets/images/lstm_architecture_detailed.png', dpi=600, bbox_inches='tight', transparent=False, facecolor='white')
plt.close()
print("Saved to assets/images/lstm_architecture_detailed.png")

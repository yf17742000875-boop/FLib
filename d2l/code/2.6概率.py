import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(10, fair_probs).sample())

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)


d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()


def bayes_theorem(prior_A: float, likelihood_B_given_A: float, evidence_B: float) -> float:
    """
    贝叶斯定理计算函数
    
    【数学原理】
    贝叶斯定理描述了如何根据新的证据更新对某个事件的信念。
    公式：P(A|B) = (P(B|A) * P(A)) / P(B)
    
    【参数说明】
    prior_A: 先验概率 P(A)
             在获取新证据之前，对事件A发生概率的初始信念
             
    likelihood_B_given_A: 似然度 P(B|A)
                         在事件A发生的条件下，观察到证据B的概率
                         
    evidence_B: 证据 P(B)
                观察到证据B的总概率（边缘概率）
    
    【返回值】
    posterior_A_given_B: 后验概率 P(A|B)
                         在观察到证据B之后，事件A发生的更新概率
    
    【应用场景】
    - 医疗诊断：根据检测结果更新患病概率
    - 垃圾邮件过滤：根据邮件内容判断是否为垃圾邮件
    - 机器学习：模型参数的后验推断
    
    【注意事项】
    - 所有输入概率值应在 [0, 1] 范围内
    - evidence_B 不能为 0，否则会导致除零错误
    """
    if not (0 <= prior_A <= 1):
        raise ValueError(f"先验概率 P(A) 必须在 [0, 1] 范围内，当前值: {prior_A}")
    if not (0 <= likelihood_B_given_A <= 1):
        raise ValueError(f"似然度 P(B|A) 必须在 [0, 1] 范围内，当前值: {likelihood_B_given_A}")
    if not (0 < evidence_B <= 1):
        raise ValueError(f"证据 P(B) 必须在 (0, 1] 范围内（不能为0），当前值: {evidence_B}")
    
    numerator = likelihood_B_given_A * prior_A
    posterior = numerator / evidence_B
    
    return posterior


def bayes_theorem_demo():
    """
    贝叶斯定理应用示例
    
    场景：医疗诊断问题
    假设某种疾病的发病率为千分之一（先验概率），
    检测该疾病的测试准确率为99%（真阳性率），
    假阳性率为1%（健康人被误诊为患病的概率）。
    
    问题：如果一个人检测结果为阳性，他实际患病的概率是多少？
    """
    print("=== 贝叶斯定理应用示例：医疗诊断 ===")
    print()
    
    P_disease = 0.001    # P(A): 先验概率，人群中患病的概率
    P_positive_given_disease = 0.99  # P(B|A): 似然度，患病者检测为阳性的概率
    P_positive_given_healthy = 0.01  # 假阳性率，健康人检测为阳性的概率
    P_healthy = 1 - P_disease  # P(not A): 人群中健康的概率
    
    P_positive = (P_positive_given_disease * P_disease) + (P_positive_given_healthy * P_healthy)
    
    print(f"已知条件：")
    print(f"  疾病发病率 P(A) = {P_disease:.4f}")
    print(f"  真阳性率 P(B|A) = {P_positive_given_disease:.2f}")
    print(f"  假阳性率 P(B|not A) = {P_positive_given_healthy:.2f}")
    print(f"  计算得到证据 P(B) = {P_positive:.4f}")
    print()
    
    P_disease_given_positive = bayes_theorem(
        prior_A=P_disease,
        likelihood_B_given_A=P_positive_given_disease,
        evidence_B=P_positive
    )
    
    print(f"计算结果：")
    print(f"  检测阳性后实际患病的概率 P(A|B) = {P_disease_given_positive:.4f}")
    print(f"  即约为 {P_disease_given_positive * 100:.2f}%")
    print()
    print("【结论】即使检测准确率高达99%，由于疾病发病率很低，")
    print("       单次阳性检测结果对应的实际患病概率仍不足10%。")


def markov_chain_joint_probability(probabilities: list) -> float:
    """
    马尔可夫链联合概率计算函数
    
    【数学原理】
    马尔可夫链的核心性质是"无后效性"：未来状态只依赖于当前状态，
    而与过去的状态无关。
    
    对于链式依赖的随机变量 A → B → C（B只依赖于A，C只依赖于B），
    联合概率可以简化为：
    P(A, B, C) = P(A) * P(B|A) * P(C|B)
    
    一般化公式（对于n个变量）：
    P(X₁, X₂, ..., Xₙ) = P(X₁) * P(X₂|X₁) * P(X₃|X₂) * ... * P(Xₙ|Xₙ₋₁)
    
    【参数说明】
    probabilities: 概率列表，按顺序包含：
                   - P(X₁): 第一个变量的先验概率
                   - P(X₂|X₁): 第二个变量依赖于第一个的条件概率
                   - P(X₃|X₂): 第三个变量依赖于第二个的条件概率
                   - ...依此类推
    
    【返回值】
    joint_prob: 联合概率 P(X₁, X₂, ..., Xₙ)
    
    【应用场景】
    - 时间序列分析：股票价格预测、天气预测
    - 自然语言处理：语言模型中的词序列概率
    - 强化学习：状态转移概率
    - 隐马尔可夫模型：语音识别、词性标注
    
    【注意事项】
    - 所有概率值必须在 [0, 1] 范围内
    - 列表至少包含1个概率值（单个事件的概率）
    """
    if len(probabilities) == 0:
        raise ValueError("概率列表不能为空")
    
    for i, prob in enumerate(probabilities):
        if not (0 <= prob <= 1):
            raise ValueError(f"第{i+1}个概率值 {prob} 必须在 [0, 1] 范围内")
    
    joint_prob = 1.0
    for prob in probabilities:
        joint_prob *= prob
    
    return joint_prob


def markov_chain_demo():
    """
    马尔可夫链联合概率计算示例
    
    场景：天气预测的马尔可夫链模型
    假设天气状态只依赖于前一天的天气：
    - 第一天晴天的概率 P(S₁) = 0.6
    - 如果第一天晴天，第二天晴天的概率 P(S₂|S₁) = 0.7
    - 如果第二天晴天，第三天下雨的概率 P(R₃|S₂) = 0.3
    
    问题：计算连续三天天气为 [晴, 晴, 雨] 的联合概率
    """
    print("=== 马尔可夫链联合概率计算示例 ===")
    print()
    
    print("【场景描述】")
    print("  天气预测的马尔可夫链模型：")
    print("  - 状态依赖：当天天气只依赖于前一天天气")
    print("  - 序列：第一天晴天 → 第二天晴天 → 第三天雨天")
    print()
    
    P_S1 = 0.6      # P(S₁): 第一天晴天的概率
    P_S2_given_S1 = 0.7  # P(S₂|S₁): 第一天晴天时第二天晴天的概率
    P_R3_given_S2 = 0.3  # P(R₃|S₂): 第二天晴天时第三天雨天的概率
    
    probabilities = [P_S1, P_S2_given_S1, P_R3_given_S2]
    joint_prob = markov_chain_joint_probability(probabilities)
    
    print("【已知概率】")
    print(f"  P(S₁) = {P_S1}")
    print(f"  P(S₂|S₁) = {P_S2_given_S1}")
    print(f"  P(R₃|S₂) = {P_R3_given_S2}")
    print()
    
    print("【联合概率公式】")
    print("  P(S₁, S₂, R₃) = P(S₁) × P(S₂|S₁) × P(R₃|S₂)")
    print(f"                = {P_S1} × {P_S2_given_S1} × {P_R3_given_S2}")
    print(f"                = {joint_prob:.4f}")
    print()
    
    print("【结论】")
    print(f"  连续三天天气为 [晴, 晴, 雨] 的概率约为 {joint_prob * 100:.2f}%")
    print()
    print("【马尔可夫性质说明】")
    print("  第三天的天气只依赖于第二天的天气，而与第一天无关。")
    print("  这就是马尔可夫链的\"无后效性\"。")


if __name__ == "__main__":
    bayes_theorem_demo()
    print()
    markov_chain_demo() 

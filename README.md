# 朱渭清的代码简介
src：这里都是重要的底层代码，定义了信道模型，通信收发端的编解码器函数，基于深度学习的信道适配器
     1.vallex_wrapper.py:我的实验不是直接拆解 VALLE-X 模型，而是在本代码中把原始 VALLE-X 的复杂接口，包装成了两个非常清晰的端————SemanticSender，SemanticReceiver
       SemanticSender:
       SemanticReceiver:

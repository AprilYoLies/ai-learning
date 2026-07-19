import numpy as np


class Tensor:
    """最简自动求导张量"""

    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)  # 梯度初始为0
        self._backward = lambda: None  # 反向传播函数
        self._prev = set(_children)  # 父节点
        self._op = _op  # 操作名称（调试用）

    # ========== 前向运算：构建计算图 ==========

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/d(self) = 1, 链式法则传递
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # d(out)/d(self) = other.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')

        def _backward():
            # d(sum(x))/dx = 1
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    # ========== 反向传播：拓扑排序 + 链式法则 ==========

    def backward(self):
        # 1. 构建拓扑序（从输出到输入）
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)  # 父节点处理完再添加自己

        build_topo(self)

        # 2. 初始化输出节点的梯度为 1（dL/dL = 1）
        self.grad = np.ones_like(self.data)

        # 3. 按逆拓扑序反向传播
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op='{self._op}')"


# ========== 测试：模拟 y = w*x + b 的反向传播 ==========

x = Tensor(2.0)  # 输入
w = Tensor(3.0)  # 权重，需要梯度
b = Tensor(4.0)  # 偏置，需要梯度

# 前向：y = w * x + b = 3*2 + 4 = 10
y = w * x + b
print(f"前向结果: {y.data}")  # 10.0

# 反向传播
y.backward()

print(f"w.grad = {w.grad}")  # dy/dw = x = 2
print(f"b.grad = {b.grad}")  # dy/db = 1
print(f"x.grad = {x.grad}")  # dy/dx = w = 3

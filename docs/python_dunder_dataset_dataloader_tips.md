# Python 小技巧：`__getitem__()` 为什么会“自动执行”？以及类方法重写(override)时到底调用谁？

读 PyTorch 的 Dataset/DataLoader 或大型工程代码时，经常会遇到两类困惑：

1) “我没手动调用 `__getitem__()`，它为什么执行了？”  
2) “明明 `__getitem__()` 在基类里，为什么最后跑的是子类的 `prepare_train_data()`？”（方法重写/动态分派）

这篇短文把两件事一次讲清楚：**魔术方法触发规则** + **继承/重写时的方法查找与调用顺序**。

---

## 1. 不是“隐函数都会自动执行”

**不会。**  
类里写的普通实例方法，只有在你显式调用时才会执行，比如 `obj.foo()`。

你感觉“自动执行”的，通常是 Python 的 **特殊方法（dunder method / 魔术方法）**，它们由解释器在特定语法触发时调用。

---

## 2. `__getitem__()` 是怎么被触发的？

当你写：

```python
x = obj[i]
```

Python 会翻译成：

```python
x = obj.__getitem__(i)
```

所以 `__getitem__()` 的“自动执行”本质是：**你触发了下标访问语法**。

---

## 3. 常见“语法 → dunder 方法”对应表

下面这些都属于“写了语法，解释器帮你调用”：

- `len(obj)` → `obj.__len__()`
- `obj[i]` → `obj.__getitem__(i)`
- `obj[i] = v` → `obj.__setitem__(i, v)`
- `for x in obj` → `obj.__iter__()`（或退化尝试 `__getitem__`）
- `obj()` → `obj.__call__()`
- `obj + other` → `obj.__add__(other)`
- `with obj:` → `obj.__enter__()` / `obj.__exit__()`

所以：  
> `__getitem__()` 不是“默认自动跑”，而是因为有人（DataLoader）在用 `dataset[i]`。

---

## 4. DataLoader 为什么会反复调用 `dataset[i]`？

训练循环通常是：

```python
for batch in dataloader:
    ...
```

`DataLoader` 的职责是不断产生 batch。要组成一个 batch，最通用方式就是：

1. `Sampler/BatchSampler` 产生一批 index：`[i1, i2, i3, ...]`
2. DataLoader 逐个取样本：`dataset[i1]`, `dataset[i2]`, ...
3. 用 `collate_fn` 把样本 list 拼成 batch
4. yield 出去

所以 `__getitem__()` 会被频繁调用：
- 每个 batch：调用约 `batch_size` 次
- 每个 epoch：调用约 `len(dataset)` 次（会受 shuffle/drop_last 等影响）

---

## 5. 重点：方法重写(override) + 动态分派到底是什么意思？

### 5.1 “重写(override)”是什么？

当 **子类** 定义了一个与 **父类同名的方法**，就会“覆盖/重写”父类版本。

```python
class Base:
    def f(self):
        print("Base.f")

class Child(Base):
    def f(self):
        print("Child.f")  # override

x = Child()
x.f()  # 输出 Child.f
```

**父类的 `f` 并不会消失，只是默认会被子类版本“遮蔽”。**

---

### 5.2 Python 调用方法时，实际调用谁？——方法查找顺序(MRO)

当你调用 `x.f()`，Python 做的是：

1. 看 `x` 的实际类型是什么（比如 `Child`）
2. 按 MRO（Method Resolution Order）在类继承链上找 `f`：
   - 先找 `Child.f`
   - 找不到再去 `Base.f`
   - 再找更上层的父类……

你可以理解为：  
> 方法是“跟着对象类型走”的，而不是“跟着你当前打开的文件走”的。

---

## 6. 最容易迷糊的一幕：基类 `__getitem__()` 调用 `self.prepare()`，结果跑到子类

这是 Dataset 工程里最常见的模式：

- 基类提供“入口流程”（例如 `__getitem__`）
- 子类只改“细节步骤”（例如 `prepare_train_data`）

### 最小例子：和 Dataset 模式完全一致

```python
class Base:
    def __getitem__(self, idx):
        print("Base.__getitem__")
        return self.prepare(idx)  # 关键：调用 self.prepare

    def prepare(self, idx):
        print("Base.prepare")
        return {"idx": idx}

class WithMap(Base):
    def prepare(self, idx):
        print("WithMap.prepare (override)")
        data = {"idx": idx}
        data["map_gt"] = "added"
        return data

ds = WithMap()
print(ds[0])
```

你会看到：

- 先执行 `Base.__getitem__`（因为子类没重写它）
- 但 `self.prepare(idx)` 会执行 `WithMap.prepare`（因为 `self` 是 `WithMap` 实例）

这就是“动态分派”：

> **即使调用点写在基类里，`self.xxx()` 也会按 `self` 的真实类型调用到子类重写的方法。**

---

## 7. 再补一个：`super()` 在重写里的作用（常用于 `__init__`）

工程里常见写法：

```python
class Child(Base):
    def __init__(self, ...):
        super().__init__(...)  # 先把父类初始化做完
        ...                    # 再做子类自己的事
```

`super()` 不是“调用某个固定父类”，而是按 MRO 找“下一个类”的实现。

因此你会经常看到：
- 子类不写 `__init__`（或 `pass`），仍会调用父类 `__init__`
- 父类里 `super().__init__()` 再继续调用更上层基类的初始化

---

## 8. 读 Dataset 代码的实用口诀

当你想搞清楚“到底执行哪段代码”：

1. **看 config 里 `dataset_type` 是谁**（对象真实类型）
2. 找这个类有没有 `__getitem__`：没有就去父类找
3. 看 `__getitem__` 里调用了哪些 `self.xxx()`：
   - 这些 `xxx` 很可能在子类被 override
4. 真正“加字段/加 GT”的逻辑，常在：
   - `prepare_train_data()` / `prepare_test_data()`
   - pipeline transforms
   - 或专门的 `_add_*()` helper

---

## 9. 总结

- `__getitem__()` 并不是“写了就会自动跑”，而是 `obj[i]` 触发的语法行为。
- `DataLoader` 为了组 batch，会反复调用 `dataset[i]`，所以你会频繁看到 `__getitem__()` 执行。
- **基类方法里调用 `self.xxx()` 时，真正执行哪个 `xxx`，由 `self` 的真实类型决定**；子类重写(override)会生效，这就是动态分派。
- `super()` 会按 MRO 调“下一个类”的实现，常用于串起父子类初始化/逻辑。

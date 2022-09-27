import os
import imp


def make_network(cfg):
    """ Import Network Module via imp.load_source() and Instantiate Network
    Urls: https://zhiqiang.org/coding/imp-load-source.html
        - imp.load_source 总是会执行模块内容, 即使重复导入和相同名字;
        - 不同的名字会生成不同的模块, 同样名字生成同样模块。不同模块不会互相影响, 指向相同模块时, 在修改其中一个变量的内部成员时, 另外一个变量也随之而变;
    Args:
        cfg.network_module - module name we want to import, 不同的 exp 可以指定导入的 module 名称
        cfg.network_path   - python file path, 去这个路径中找要导入的 module
    """
    module = cfg.network_module
    path = cfg.network_path
    network = imp.load_source(module, path).Network()
    return network

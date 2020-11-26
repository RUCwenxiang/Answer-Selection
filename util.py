def gbk_to_utf8(source, target):
    with open(source, "r", encoding="gbk") as src, \
        open(target, "w", encoding="utf-8") as dst:
            for line in src.readlines():
                dst.write(line)
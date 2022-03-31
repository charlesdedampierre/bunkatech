def erase_comma(x):
    try:
        res = x.split(",")[0]
    except:
        res = x
    return res


def erase_point(x):
    try:
        res = x.split(".")[0]
    except:
        res = x
    return res


def erase_accent(x):
    try:
        res = x.split("'")[1]
    except:
        res = x
    return res

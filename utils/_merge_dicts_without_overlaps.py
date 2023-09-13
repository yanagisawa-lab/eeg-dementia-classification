import mngs

def merge_dicts_without_overlaps(*confs):
    merged_conf = {}
    for c in confs:
        assert mngs.general.search(merged_conf.keys(), c.keys()) == ([], [])
        merged_conf.update(c)
    return merged_conf

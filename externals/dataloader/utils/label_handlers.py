#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-09-14 10:30:01 (ywatanabe)"

def _expand_conc_classes_str(conc_classes, delimiter="+"):
    """
    Example:
        _expand_conc_classes_str(["HV", 'AD+DLB+NPH'])
        # [["HV"], ["AD"], ["DLB"], ["NPH"]]
    """
    expanded_classes = []
    for k in conc_classes:
        if delimiter in k:
            expanded_classes.append(k.split(delimiter))
        else:
            expanded_classes.append([k])
    return expanded_classes


def _mk_dict_for_conc_class_str_2_label_int(conc_classes, delimiter="+"):
    """
    Example:
        conc_classes = ['HV', 'AD+DLB+NPH']
        _mk_dict_for_conc_class_str_2_label_int(conc_classes, delimiter="+")
        # {'HV': 0, 'AD': 1, 'DLB': 1, 'NPH': 1}
    """
    class2label_dict = {}
    expanded_classes = _expand_conc_classes_str(conc_classes, delimiter)
    for i, class_names in enumerate(expanded_classes):
        for class_name in class_names:
            assert class_name not in class2label_dict  # 既にclass_nameが登録済みの場合はエラー
            class2label_dict[class_name] = i
    return class2label_dict


def _mk_dict_for_label_int_2_conc_class(conc_classes):
    """
    Example:
        conc_classes = ['HV', 'AD+DLB+NPH']
        _mk_dict_for_label_int_2_conc_class(conc_classes)
        # {0: 'HV', 1: 'AD+DLB+NPH'}
    """
    label2class_dict = {}
    for i, conc_class in enumerate(conc_classes):
        label2class_dict[i] = conc_class
    return label2class_dict

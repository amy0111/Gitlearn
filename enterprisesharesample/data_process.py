# -*- encoding: utf-8 -*-
"""

股权穿透demo

处理原始csv文件,使其格式符合neo4j-admin import数据导入工具。
原始数据文件存放在data_sample。
经过处理的数据文件存放在neo4j_import_data中。

"""

import pandas as pd
import os


def data_process():
    save_path = "neo4j_import_data"
    try:
        os.mkdir(save_path)
    except FileExistsError as e:
        print(e)

    # CSV header format
    # 实体：
    # person_id:ID,name,:LABEL (Person)
    node_person_path = r"data_sample/person.csv"
    person_df = pd.read_csv(node_person_path, header=None, names=["person_id:ID", "name"])
    person_df[":LABEL"] = "Person"
    person_df.to_csv(os.path.join(save_path, node_person_path.split("/")[-1]), index=False)

    # corp_id:ID,name,:LABEL (Corp)
    node_corp_path = r"data_sample/corp.csv"
    corp_df = pd.read_csv(node_corp_path, header=None, names=["corp_id:ID", "name"])
    corp_df[":LABEL"] = "Corp"
    corp_df.to_csv(os.path.join(save_path, node_corp_path.split("/")[-1]), index=False)

    # 关系：
    # :START_ID,degree:float,:END_ID,:TYPE Person-[RELATIVE_WITH {degree}]-Person
    person_rel_path = r"data_sample/person_rel.csv"
    person_rel_df = pd.read_csv(person_rel_path, header=None, names=[":START_ID", ":END_ID", "degree:float"])
    person_rel_df[":TYPE"] = "RELATIVE_WITH"
    person_rel_df.to_csv(os.path.join(save_path, person_rel_path.split("/")[-1]), index=False)

    # :START_ID,share:float,:END_ID,:TYPE Person-[HOLD_SHARE {share}]-Corp
    person_corp_share_path = r"data_sample/person_corp_share.csv"
    person_corp_share_df = pd.read_csv(person_corp_share_path, header=None,
                                       names=[":START_ID", ":END_ID", "share:float"])
    person_corp_share_df[":TYPE"] = "HOLD_SHARE"
    person_corp_share_df.to_csv(os.path.join(save_path, person_corp_share_path.split("/")[-1]), index=False)

    # :START_ID,role,:END_ID,:TYPE Person-[ROLE_AS {role}]-Corp
    person_corp_role_path = r"data_sample/person_corp_role.csv"
    person_corp_role_df = pd.read_csv(person_corp_role_path, header=None, names=[":START_ID", ":END_ID", "role"])
    person_corp_role_df[":TYPE"] = "ROLE_AS"
    person_corp_role_df.to_csv(os.path.join(save_path, person_corp_role_path.split("/")[-1]), index=False)

    # :START_ID,share:float,:END_ID,:TYPE Corp-[HOLD_SHARE {share}]-Corp
    corp_share_path = r"data_sample/corp_share.csv"
    corp_share_df = pd.read_csv(corp_share_path, header=None, names=[":START_ID", ":END_ID", "share:float"])
    corp_share_df[":TYPE"] = "HOLD_SHARE"
    corp_share_df.to_csv(os.path.join(save_path, corp_share_path.split("/")[-1]), index=False)

    # :START_ID,:END_ID,:TYPE Corp-[IS_BRANCH_OF]-Corp
    corp_rel_path = r"data_sample/corp_rel.csv"
    corp_rel_df = pd.read_csv(corp_rel_path, header=None, names=[":START_ID", ":END_ID"])
    corp_rel_df[":TYPE"] = "IS_BRANCH_OF"
    corp_rel_df.to_csv(os.path.join(save_path, corp_rel_path.split("/")[-1]), index=False)

    print("Done!")



if __name__ == '__main__':
    data_process()

import itertools

import pandas as pd

from package_file_conversion.poscar2df import poscar2df
from package_file_conversion.nnlist2df import nnlist2df
from package_bond_search_algorithms.algolithm_bond_search_for_trigonal_pyramidal_planar_shape import concat_filter
from package_file_conversion.df2poscar import df2poscar


def mk_polyatomic_ion_replaced_point_poscar(poscar_path,
                                            nnlist_path,
                                            central_atom_symbol='C',
                                            neighboring_atom_symbol='O',
                                            bond_length_lower_end=0.99,
                                            bond_length_upper_end=1.66,
                                            generated_poscar_path='./ion_replaced_point/POSCAR'):
    """
    This func() makes a new POSCAR file which polyatomic ion replaced a point.

    Usage:
    ------
    mk_polyatomic_ion_replaced_point_poscar(poscar_path=poscar_path,
                                            nnlist_path=nnlist_path,
                                            central_atom_symbol=central_atom_symbol,
                                            neighboring_atom_symbol=neighboring_atom_symbol,
                                            bond_length_lower_end=bond_length_lower_end,
                                            bond_length_upper_end=bond_length_upper_end,
                                            generated_poscar_path=generated_poscar_path)

    Parameters:
    -----------
    poscar_path: str or pathlib.Path
    nnlist_path: str or pathlib.Path
    central_atom_symbol: str
    neighboring_atom_symbol: str
    bond_length_lower_end: str or float
    bond_length_upper_end: str or float
    generated_poscar_path: str or pathlib.Path

    Return:
    -------
    None
    """
    # 0-1. POSCAR, POSCAR.nnlistをDataFrameに変換する
    df_poscar = poscar2df(poscar_path=poscar_path)
    df_nnlist = nnlist2df(nnlist_path=nnlist_path)

    # 0-2. 多原子イオンを含むかどうかの判定フィルター関数を実行
    central_atom_symbol = central_atom_symbol
    neighboring_atom_symbol = neighboring_atom_symbol
    bond_length_lower_end = float(bond_length_lower_end)
    bond_length_upper_end = float(bond_length_upper_end)
    bool_, ion_central_atom_ids = concat_filter(df_nnlist=df_nnlist,
                                                central_atom_symbol=central_atom_symbol,
                                                neighboring_atom_symbol=neighboring_atom_symbol,
                                                bond_length_lower_end=bond_length_lower_end,
                                                bond_length_upper_end=bond_length_upper_end)

    if bool_:
        # 1. 多原子イオンを点で置換した絶対座標のDataFrameを作成
        atom_ids_belonging_to_polyatomic_ions = []
        for ion_central_atom_id in ion_central_atom_ids:
            df_nnlist_ion_central_atom_id_filterd = df_nnlist[df_nnlist['central_atom_id'] == ion_central_atom_id]
            df_nnlist_ion_central_atom_id_filterd_sorted = df_nnlist_ion_central_atom_id_filterd.sort_values('rel_distance')
            df_nnlist_rel_distance_filter = df_nnlist_ion_central_atom_id_filterd_sorted['rel_distance'] < bond_length_upper_end
            df_nnlist_rel_distance_filterd = df_nnlist_ion_central_atom_id_filterd.sort_values('rel_distance')[df_nnlist_rel_distance_filter]
            # 'neighboring_atom_id'カラムをリストとして取得
            neighboring_atom_ids = df_nnlist_rel_distance_filterd['neighboring_atom_id'].tolist()
            atom_ids_belonging_to_polyatomic_ions.append(neighboring_atom_ids)
        # 2重リストを1重リストに変換
        atom_ids_belonging_to_polyatomic_ions = list(itertools.chain.from_iterable(atom_ids_belonging_to_polyatomic_ions))
        # 数字の順番に並べ替え
        atom_ids_belonging_to_polyatomic_ions = sorted(atom_ids_belonging_to_polyatomic_ions, key=lambda s: int(s))
        # df_poscarから，多原子イオンに属す原子を抽出するフィルターを作成
        atom_ids_belonging_to_polyatomic_ions_filter = df_poscar['atom_id'].apply(lambda s: s in atom_ids_belonging_to_polyatomic_ions)
        # df_poscarから，多原子イオンに属さない原子を抽出するフィルターを作成
        atom_ids_not_belonging_to_polyatomic_ions_filter = ~atom_ids_belonging_to_polyatomic_ions_filter
        # df_poscarから，多原子イオンに属しかつ中心原子となる原子を抽出するフィルターを作成
        atom_ids_that_polyatomic_ions_center_filter = df_poscar['atom_id'].apply(lambda s: s in ion_central_atom_ids)
        # フィルターを結合し，（多原子イオンに含まれない）または（多原子イオンに含まれかつ多原子イオンの中心）となる行を抽出するフィルターを作成
        new_poscar_atom_ids_filter = atom_ids_not_belonging_to_polyatomic_ions_filter | atom_ids_that_polyatomic_ions_center_filter
        # 作成したフィルターを適用し，多原子イオンを点で置換した絶対座標のDataFrameを作成
        df_poscar_abs_coords = df_poscar[new_poscar_atom_ids_filter]

        # 2. 多原子イオンの相対中心座標のDataFrameを作成
        df_nnlist_rel_coords_series_list = []
        for ion_central_atom_id in ion_central_atom_ids:
            df_nnlist_ion_central_atom_id_filterd = df_nnlist[df_nnlist['central_atom_id'] == ion_central_atom_id]
            df_nnlist_ion_central_atom_id_filterd_sorted = df_nnlist_ion_central_atom_id_filterd.sort_values('rel_distance')
            df_nnlist_rel_distance_filter = df_nnlist_ion_central_atom_id_filterd_sorted['rel_distance'] < bond_length_upper_end
            df_nnlist_rel_distance_filterd = df_nnlist_ion_central_atom_id_filterd.sort_values('rel_distance')[df_nnlist_rel_distance_filter]
            # rel_x, re_y, re_zごとに平均をとる
            df_nnlist_rel_distance_filterd_cols_dropped = df_nnlist_rel_distance_filterd[['central_atom_id', 'rel_x', 'rel_y', 'rel_z']]
            # 'central_atom_symbol'カラムでgroupbyしmeanを計算した後，groupbyed列(:'central_atom_symbol'カラム)をカラムにする
            df_nnlist_rel_distance_filterd_cols_dropped_meaned = df_nnlist_rel_distance_filterd_cols_dropped.groupby('central_atom_id').mean().reset_index()
            df_nnlist_rel_coords_series_list.append(df_nnlist_rel_distance_filterd_cols_dropped_meaned)
        # df_nnlist_rel_coords_series_listのSeriesを文字列化して，df_poscarと同じ形式のDataFrameに整形する
        df_nnlist_rel_coords_list = [str(s).split(' ')[-4:] for s in df_nnlist_rel_coords_series_list]
        df_nnlist_rel_coords = pd.DataFrame(df_nnlist_rel_coords_list, columns=['central_atom_id', 'rel_x', 'rel_y', 'rel_z'])
        # 'rel_x', 'rel_y', 'rel_z'カラムをstr型からfloat型に変換
        df_nnlist_rel_coords[['rel_x', 'rel_y', 'rel_z']] = df_nnlist_rel_coords[['rel_x', 'rel_y', 'rel_z']].astype(float)

        # 3. 1.で得たdf_poscar_abs_coordsと2.で得たdf_nnlist_rel_coordsを足し合わせる
        # 足し合わせ計算の便宜上，2つのDataFrameを1つのDataFrameに結合する
        df_nnlist_poscar_merged = pd.merge(df_poscar_abs_coords, df_nnlist_rel_coords, left_on='atom_id', right_on='central_atom_id', how='left')
        df_nnlist_poscar_merged[['rel_x', 'rel_y', 'rel_z']] = df_nnlist_poscar_merged[['rel_x', 'rel_y', 'rel_z']].fillna(0)
        # 多原子イオンの絶対中心座標と相対座標を足し合わせる
        df_nnlist_poscar_merged['x'] = df_nnlist_poscar_merged['x'] + df_nnlist_poscar_merged['rel_x']
        df_nnlist_poscar_merged['y'] = df_nnlist_poscar_merged['y'] + df_nnlist_poscar_merged['rel_y']
        df_nnlist_poscar_merged['z'] = df_nnlist_poscar_merged['z'] + df_nnlist_poscar_merged['rel_z']
        # 多原子イオンの絶対中心座標と相対座標の足し合わせのために便宜上用意した，不要なカラムを削除
        df_poscar_ion_replaced_point = df_nnlist_poscar_merged.drop(columns=['central_atom_id', 'rel_x', 'rel_y', 'rel_z'])

        # 4. 3.で生成したdf_poscar_ion_replaced_pointをdf2poscar()を用いてPOSCARファイルとして書き出す
        df2poscar(df_poscar_ion_replaced_point, original_poscar_path=poscar_path, generated_poscar_path=generated_poscar_path)

    else:
        pass


if __name__ == "__main__":
    poscar_path = 'sample_test_files/POSCAR'
    nnlist_path = 'sample_test_files/POSCAR.nnlist'
    central_atom_symbol = 'C'
    neighboring_atom_symbol = 'O'
    bond_length_lower_end = 0.99
    bond_length_upper_end = 1.66
    generated_poscar_path = 'sample_test_files/gen_data/POSCAR'
    mk_polyatomic_ion_replaced_point_poscar(poscar_path=poscar_path,
                                            nnlist_path=nnlist_path,
                                            central_atom_symbol=central_atom_symbol,
                                            neighboring_atom_symbol=neighboring_atom_symbol,
                                            bond_length_lower_end=bond_length_lower_end,
                                            bond_length_upper_end=bond_length_upper_end,
                                            generated_poscar_path=generated_poscar_path)

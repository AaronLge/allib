import numpy as np
from allib import general as gl
import scipy as sc
import pandas as pd
import sys
import sqlite3
from sqlite3 import Error
import os
import shutil
import chardet

path = r"C:\\temp\\python_self_crated\\packages"
sys.path.insert(0, path)
from allib import general as gl


# %%classes
class Segment:
    def __init__(self, num, angles, angle_name, result, indizes=None, colnames=None):
        self.num = num
        self.angles = angles
        self.result = result
        self.indizes = indizes
        if angles is not None:
            self.angles_mod = [angles[0] % 360, angles[1] % 360]
        else:
            self.angles_mod = None

        self.colnames = colnames
        if colnames is None:
            self.colnames = dict()

        if indizes is not None:
            self.N = len(indizes)
        else:
            self.N = None
        self.angle_name = angle_name

    def FilterData(self, Data):
        Data = Data.loc[(Data[self.angle_name] >= self.angles_mod[0])
                        | (Data[self.angle_name] < self.angles_mod[1])]
        return Data


class Calculation:

    def __init__(self, result=None, basedata=None, filt=None):
        self.result = result

        self.basedata = basedata
        if basedata is None:
            self.basedata = dict()

        self.filt = filt

    def initilize_from_db(self, db_path, table_name, colnames, **kwargs):
        """initilizes dataframe from sql database at "db_path" by loading the colims specified by colnames in the table "table_name". If colnames is None, all columns are loaded
         Returns dataframe

         optional:
         timeframe: list of two datetime objects specifying the start and end time of the data to be evaluated
         """

        timeframe = kwargs.get('timeframe', None)
        indizes = kwargs.get('indizes', None)

        df = gl.export_df_from_sql(db_path, table_name, column_names=colnames)
        df.index = pd.to_datetime(df.index)

        if timeframe is not None:
            df = df.loc[timeframe[0]:timeframe[1]]

        self.basedata["dbname"] = db_path
        self.basedata["tablename"] = table_name
        self.basedata["colnames_ini"] = colnames
        self.basedata["dbdate"] = None
        self.basedata["db_timeframe"] = [df.index[0], df.index[-1]]
        self.basedata["N_rows"] = len(df)
        self.basedata["sample_rate"] = df.index[1] - df.index[0]

        if indizes is None:
            self.basedata["indizes"] = df.index
        else:
            self.basedata["indizes"] = indizes

        return df

    def initilize_filter(self, colnames, mode='range', ranges=None):
        """filters data specified in "basedata" propterty, has to be initilized by using "initilize_from_db". creats "filt" dictionary, that contains:
         indizes_in: datetime index object, with all indizies still in the filtered list
         indizes_out: datetime index object, with all indizies excluded by filtering
         colnames: colnames
         ranges: ranges

        INPUT:
        colnames: list of stings, columnames to filter by in Dataframe
        ranges: list of lists, list containing len(colnames) elements of length 2, in fomrmat [a_min, a_max] specifing the range of the filtered data
                if a_max is None, maximal value of the correspnding values of column spcified by colname in with same index used

        retrun:
        indizes: datetime index object, with all indizies still in the filtered list"""

        indizes_full = self.basedata["indizes"]

        if colnames is None:
            df_filt = gl.export_df_from_sql(self.basedata["dbname"], self.basedata["tablename"])
            colnames = list(df_filt.columns)
        else:
            df_filt = gl.export_df_from_sql(self.basedata["dbname"], self.basedata["tablename"], column_names=colnames)

        self.filt = dict()

        if mode == 'range':
            for i, range_curr in enumerate(ranges):
                if range_curr[1] is None:
                    range_curr[1] = np.max(df_filt[colnames[i]])

            a_min = [curr[0] for curr in ranges]
            a_max = [curr[1] for curr in ranges]

            df_filt = gl.filter_dataframe(df_filt, colnames, a_min, a_max)
            self.filt["ranges"] = ranges
            self.filt["colnames"] = colnames

        if mode == 'nans':
            df_filt = df_filt.dropna(how='any')

        index_filt = df_filt.index
        indizes_in = index_filt.intersection(indizes_full)
        indizes_out = indizes_full.difference(index_filt)

        self.filt["indizes_in"] = indizes_in
        self.filt["indizes_out"] = indizes_out
        self.filt["mode"] = mode

        return indizes_in
    
    def create_segment_title(self, mode='verbose', latex=True):
        """"if the data stored in result is list of "Segment",  it exports a list of title in Latex format
            if mode='general', Segment.basedata has to be initilized
        """""

        titles = []
        for segment_curr in self.result:
            underscore = str()
            header = str()
            N_exp = segment_curr.N

            if segment_curr.angle_name is not None:
                if latex:
                    header += r"\small " f"'{segment_curr.angle_name}': {segment_curr.angles[0]}° to {segment_curr.angles[1]}°"
                else:
                    header += f"'{segment_curr.angle_name}':{segment_curr.angles[0]}° to {segment_curr.angles[1]}°"

            else:
                header += "omnidirectional"

            if mode == "verbose":
                N_ges = self.basedata["N_rows"]
                timeframe = self.basedata["db_timeframe"]
                sample_rate = self.basedata["sample_rate"]
                if latex:
                    underscore = (r"\scriptsize " + f"samples: {N_exp:.2e} ({round(N_exp / N_ges * 100, 1)}\%), " +
                                  f"{timeframe[0].round('1d').date()} to {timeframe[1].round('1d').date()}, " +
                                  f"d_t: {sample_rate.total_seconds()} s")
                else:
                    underscore = (f"samples: {N_exp:.2e} ({round(N_exp / N_ges * 100, 1)}%), " +
                                  f"timeframe: {timeframe[0].round('1d').date()} to {timeframe[1].round('1d').date()}, " +
                                  f"time step: {sample_rate.total_seconds()} s")

            elif mode == "standard":
                N_ges = self.basedata["N_rows"]

                if latex:
                    underscore = r"\scriptsize " + f"samples: {N_exp:.2e} ({round(N_exp / N_ges * 100, 1)}%)"
                else:
                    underscore = f"samples: {N_exp:.2e} ({round(N_exp / N_ges * 100, 1)}%)"

            elif mode == "sparse":
                underscore = None

            else:
                underscore = None

            if underscore is not None:
                titles.append(header + "\n" + underscore)
            else:
                titles.append(header)

        return titles

    def load_from_db(self, column_names=None, applie_filt=True, colnames_ini=False, **kwargs):
        """wrapper for export_df_from_sql in general lib, takes db_name and tablename from Calculation information, applies filter if it is there"""
        if colnames_ini:
            column_names = self.basedata["colnames_ini"]

        df = gl.export_df_from_sql(self.basedata["dbname"], self.basedata["tablename"], column_names=column_names)

        if (self.filt is None) or not applie_filt:
            df = df[df.index.isin(self.basedata["indizes"])]
        else:
            df = df[df.index.isin(self.filt["indizes_in"])]

        return df


class DataCol:
    def __init__(self, name_data=str(), name_plot=str(), db_name=None, table_raw=None, symbol=None):
        self.name_data = name_data
        self.name_plot = name_plot
        self.db_name = db_name

        if table_raw is None:
            self.table_raw = []

        self.symbol = symbol


# %% general functions

def percentiles(df, percent: list):
    for n_perc, perc in enumerate(percent):
        z = sc.stats.norm.ppf(perc / 100)

        df[f'{perc}th percentile'] = df["mean"] + z * df["std"]
    return


def condensation(x, y, grid, **kwargs):
    # todo: berschreibung!
    reg_model = kwargs.get('reg_model', 'poly')
    deg_reg = kwargs.get('deg_reg', 3)
    cut_reg = kwargs.get('cut_reg', 0)
    reg_weighting = kwargs.get('reg_weighting', 0)
    line_plot_zone = kwargs.get('line_plot_zone', [None, None])
    reg_zone = kwargs.get('reg_zone', [None, None])
    perc = kwargs.get('percentiles', [])
    bin_min = kwargs.get('bin_min', 0)
    perc_mean = kwargs.get('perc_mean', 50)
    avrg_method = kwargs.get('avrg_method', 'mean')
    make_monotone = kwargs.get("make_monotone", False)

    if reg_zone[1] is None:
        reg_zone[1] = max(x)

    if reg_zone[0] is None:
        reg_zone[0] = min(x)

    if line_plot_zone[1] is None:
        line_plot_zone[1] = max(x)

    if line_plot_zone[0] is None:
        line_plot_zone[0] = min(x)

    n_bin = len(grid) - 1
    x_zone = [min(grid), max(grid)]

    averaged, std, count = gl.grid_pointcloud_in_x(
        x, y, grid, method=avrg_method)

    OUT = pd.DataFrame()
    OUT["x"] = averaged.index
    OUT["mean"] = averaged.values
    OUT["std"] = std.values
    OUT["count"] = count.values

    if perc_mean != 50:
        z = sc.stats.norm.ppf(perc_mean / 100)
        OUT['mean'] = OUT['mean'] + z * OUT['std']

    if make_monotone:
        OUT['mean'] = gl.make_monotone(OUT['mean'])

    OUT['isData'] = 1
    OUT.loc[OUT['count'] <= bin_min, 'isData'] = 0

    nanMask = np.array(OUT['isData'])

    # 95% Grenze
    N_upper = round(cut_reg / 100 * len(x))

    OUT['bool_upper'] = 0

    x_points_sorted = np.sort(x)

    if N_upper == 0:
        OUT.loc[OUT.index[:], 'bool_upper'] = 1
    elif N_upper != len(x):
        vm_lim_upper = min(x_points_sorted[N_upper:-1])

        _, edges, vs_bin = sc.stats.binned_statistic(
            vm_lim_upper, vm_lim_upper, statistic='count', bins=n_bin, range=x_zone)

        OUT.loc[OUT.index[vs_bin[0]:], 'bool_upper'] = 1

    # mindesdatenpunkte in Bin
    if len(perc) > 0:
        percentiles(OUT, percent=perc)

    # regression
    # regressionsbereich
    OUT['bool_reg_zone'] = 0

    OUT.loc[(OUT['x'] > reg_zone[0]) & (
            OUT['x'] < reg_zone[1]), 'bool_reg_zone'] = 1

    # regression plotbereich
    OUT['bool_reg_plot'] = 0

    if line_plot_zone[1] is None:

        nanMask_temp = nanMask.copy()
        nanMask_temp[min(np.where(nanMask == 1)[0]):max(np.where(nanMask == 1)[0])] = 1

        OUT.loc[(OUT['x'] > line_plot_zone[0]) & (
                nanMask_temp == 1), 'bool_reg_plot'] = 1
    else:
        OUT.loc[(OUT['x'] > line_plot_zone[0]) & (
                OUT['x'] < line_plot_zone[1]), 'bool_reg_plot'] = 1

    col_key = [
        col for col in OUT.columns if 'mean' in col or 'percentile' in col]

    # mask with bereich
    x_reg_zone = OUT.loc[(OUT['bool_reg_zone'] == 1) & (nanMask == 1), 'x']

    weights = OUT.loc[(OUT['bool_reg_zone'] == 1) & (
            nanMask == 1), 'count']

    for name in col_key:
        if len(x_reg_zone) != 0:

            y_reg_zone = OUT.loc[(OUT['bool_reg_zone'] == 1) & (nanMask == 1), name]

            reg_model = gl.model_regression(x_reg_zone, y_reg_zone, degree=deg_reg, weights=weights, weights_regulation=reg_weighting, reg_model=reg_model)

            OUT[f'{name} regression'] = float('nan')

            pred_reg = gl.predict_regression(reg_model, OUT['x'])

            OUT[f'{name} regression'] = pred_reg

        else:
            OUT[f'{name} regression'] = float('nan')

        OUT[f'{name} result'] = OUT[name]

        OUT.loc[OUT['bool_upper'] == 1, f'{name} result'] = OUT.loc[OUT['bool_upper'] == 1, f'{name} regression']

        OUT[f'{name} result plot'] = float('nan')

        OUT.loc[OUT['bool_reg_plot'] == 1, f'{name} result plot'] = OUT.loc[OUT['bool_reg_plot'] == 1, f'{name} result']

    return OUT


def quantiles(perc_low, middle, perc_up, quant_low, quant_up):
    """" expects 2 percentiles and HSTP Data """
    quantile = np.empty(len(middle))
    quantile[:] = float('nan')

    try:
        T_up = 1 / quant_up
        T_low = 1 / quant_low

        bool_LOW = (perc_up < T_low)
        bool_MIDDLE = (middle >= T_low) & (middle <= T_up)

        bool_UP = (perc_low > T_up)

        bool_CONNECT_LOW = ~bool_LOW & ~bool_MIDDLE & ~bool_UP & (middle < (T_low + T_up) / 2)
        bool_CONNECT_HIGHT = ~bool_LOW & ~bool_MIDDLE & ~bool_UP & (middle > (T_low + T_up) / 2)

        quantile[bool_CONNECT_LOW] = T_low
        quantile[bool_CONNECT_HIGHT] = T_up

        quantile[bool_LOW] = perc_up[bool_LOW]

        quantile[bool_MIDDLE] = middle[bool_MIDDLE]
        quantile[bool_UP] = perc_low[bool_UP]

    except:
        print(
            f"    quantile not possible for segment, check if graph is monotone in 'line_plot_zone' or if percentiles cross in the frequency band. quantile is set to mean")

    return quantile


# def MetOcan_to_sqlDB(Paths, db_name, resample_rate):
#     """accepts Dict of csv paths (Metocean Format!) and stores the individual data as well a resampled version in
#     a combined table to a sql database. If the database already exists, it will be overwritten
#
#     Parameters:
#         Paths: dict with the keys: {path_wave, path_atmo, path_ocean} and the corresponding paths to the csv files, can be None
#         db_name: sting to name the database, with path if necessary
#         resample_rate: resample rate in the fomrat: float {y,d,s,m} for year, day, second, month. Example: "1d"
#
#     Return:
#         sql Database at the desired path
#     """
#
#     def join_dataframes_by_index(dfs_all):
#         # Filter out empty dataframes
#         dataframes = [df for df in dfs_all if not df.empty]
#
#         # Perform the join operation on the non-empty dataframes
#         if dataframes:
#             res = dataframes[0]
#             for df in dataframes[1:]:
#                 res = res.join(df, how='inner')
#             return res
#         else:
#             return pd.DataFrame()  # Return
#
#     def read_MetaData(file):
#         global name
#         Names = []
#         Values = []
#         with open(file) as input_file:
#             for _ in range(14):
#                 line = input_file.readline()
#                 line = line.replace('\n', '')
#                 line = line.replace('"', '')
#                 splited = line.split('\t')
#                 name = splited[0]
#                 value = splited[-1][2:]
#                 Names.append(name)
#                 Values.append(value)
#
#         return Names, Values
#
#     NAMES = []
#     VALUES = []
#     db_path = os.path.dirname(Paths[list(Paths.keys())[0]]) + "/" + db_name
#     db_exists = os.path.exists(db_path)
#
#     if db_exists:
#         overwrite = input(f"    Database {db_path} already exists, overwrite? (y/n)")
#         if overwrite == 'y':
#             os.remove(db_path)
#         else:
#             return None
#
#     conn = sqlite3.connect(db_path)
#
#     if Paths["path_wave"] is not None:
#         print("    wave data found")
#
#         df_wave_NAN = pd.read_csv(Paths["path_wave"], skiprows=15, na_filter=False)
#         df_wave = df_wave_NAN.dropna(how='any')
#         df_wave.set_index('datetime (ISO 8601) [UTC]', inplace=True)
#         df_wave.index = pd.to_datetime(df_wave.index)
#
#         temp = read_MetaData(Paths["path_wave"])
#         temp[0].append("NANs")
#         temp[1].append(len(df_wave_NAN) - len(df_wave))
#         temp[0].insert(0, "DataSet")
#         temp[1].insert(0, "Waves Data")
#         NAMES.append(temp[0])
#         VALUES.append(temp[1])
#
#         df_wave.to_sql('Waves', conn)
#
#         df_wave_resample = df_wave.resample(resample_rate).mean()
#
#     else:
#         df_wave_resample = pd.DataFrame()
#         print("    no wave data found")
#
#     if Paths["path_atmo"] is not None:
#         print('    atmospheric data found')
#
#         df_wind_NAN = pd.read_csv(Paths["path_atmo"], skiprows=15, na_filter=False)
#         df_wind = df_wind_NAN.dropna(how='any')
#         df_wind.set_index('datetime (ISO 8601) [UTC]', inplace=True)
#         df_wind.index = pd.to_datetime(df_wind.index)
#
#         temp = read_MetaData(Paths["path_atmo"])
#         temp[0].append("NANs")
#         temp[1].append(len(df_wave_NAN) - len(df_wave))
#         temp[0].insert(0, "DataSet")
#         temp[1].insert(0, "Athmospheric Data")
#         NAMES.append(temp[0])
#         VALUES.append(temp[1])
#
#         df_wind.to_sql('Athmosphere', conn)
#
#         df_wind_resample = df_wind.resample(resample_rate).mean()
#
#     else:
#         df_wind_resample = pd.DataFrame()
#         print("    no Athmospheric Data data found")
#
#     if Paths["path_ocean"] is not None:
#         print('    oceanic data found')
#         df_water_NAN = pd.read_csv(Paths["path_ocean"], skiprows=15, na_filter=False)
#         df_water = df_water_NAN.dropna(how='any')
#         df_water.set_index('datetime (ISO 8601) [UTC]', inplace=True)
#         df_water.index = pd.to_datetime(df_water.index)
#         temp = read_MetaData(Paths["path_ocean"])
#         temp[0].append("NANs")
#         temp[1].append(len(df_wave_NAN) - len(df_wave))
#         temp[0].insert(0, "DataSet")
#         temp[1].insert(0, "Ocean Data")
#         NAMES.append(temp[0])
#         VALUES.append(temp[1])
#
#         df_water.to_sql('Ocean', conn)
#         df_water_resample = df_water.resample(resample_rate).mean()
#     else:
#         df_water_resample = pd.DataFrame()
#         print("    no oceanic data found")
#
#     df_ges = join_dataframes_by_index([df_wave_resample, df_wind_resample, df_water_resample])
#
#     df_ges.to_sql('Combined', conn)
#
#     # Metadata
#
#     df_Meta = pd.DataFrame(columns=NAMES[0][1:], index=[values[0] for values in VALUES])
#
#     for values in VALUES:
#         df_Meta.loc[values[0], :] = values[1:]
#
#     df_Meta.to_sql('MetaData', conn)
#
#     conn.close()
#
#     return db_path


def csv_to_sqlDB(path_csvs, db_name, resample_rate, data_kind='MetOcean', encoding='auto', nans=None, skiprows=None, delimiter=';', dayfirst=False, datetime_mode='single_col', low_memory=True, drop_rows=None):
    """accepts Dict of csv paths (Metocean Format!) and stores the individual data as well a resampled version in
    a combined table to a sql database. If the database already exists, it will be overwritten

    Parameters:
        Paths: dict with the keys: {path_wave, path_atmo, path_ocean} and the corresponding paths to the csv files, can be None
        db_name: sting to name the database, with path if necessary
        resample_rate: resample rate in the fomrat: float {y,d,s,m} for year, day, second, month. Example: "1d"

    Return:
        sql Database at the desired path
    """

    def join_dataframes_by_index(dataframes):
        # Perform the join operation on the non-empty dataframes
        res = dataframes[0]
        for df in dataframes[1:]:
            res = res.join(df, how='inner')
        return res

    def read_MetaData_Metocean(file):
        global name
        Names = []
        Values = []
        with open(file) as input_file:
            for _ in range(14):
                line = input_file.readline()
                line = line.replace('\n', '')
                line = line.replace('"', '')
                splited = line.split('\t')
                name = splited[0]
                value = splited[-1][2:]
                Names.append(name)
                Values.append(value)

        return Names, Values

    def check_encoding(path, sample_size=10000):
        with open(path, 'rb') as file:
            # Read a sample of the file
            raw_data = file.read(sample_size)
            # Detect the encoding
            result = chardet.detect(raw_data)
            encoding = result
            return encoding

    data_resampled = []
    NAMES = []
    VALUES = []
    csv_files = [os.path.join(path_csvs, f) for f in os.listdir(path_csvs) if f.endswith(('.csv', '.CSV'))]

    db_path = path_csvs + "/" + db_name
    db_exists = os.path.exists(db_path)

    if db_exists:
        overwrite = input(f"    Database {db_path} already exists, overwrite? (y/n)")
        if overwrite == 'y':
            os.remove(db_path)
        else:
            return None

    conn = sqlite3.connect(db_path)

    for file in csv_files:
        
        if encoding == 'auto':
            encoding = check_encoding(file)
    
        file_name = os.path.basename(file)
        print(f"adding {file_name} to database")

        if data_kind == 'MetOcean':
            # values
            skiprows = 15
            df_NAN = pd.read_csv(file, skiprows=skiprows, na_filter=False, encoding=encoding)

            # metadata
            meta = read_MetaData_Metocean(file)

        elif data_kind == 'APGMer':
            # values
            df_NAN = pd.read_csv(file, sep=';', low_memory=False, na_filter=False, encoding=encoding)
            df_NAN = df_NAN.drop(0)
            df_datetime = df_NAN[df_NAN.columns[[0, 1, 2, 3, 4]]]

            df_NAN.drop(df_NAN.columns[[0, 1, 2, 3, 4]], axis=1, inplace=True)

            # metadata
            meta = [[], []]
        
        elif data_kind is None:
            df_NAN = pd.read_csv(file, skiprows=skiprows, encoding=encoding, delimiter=delimiter, na_values=nans, low_memory=low_memory)

            if datetime_mode == 'single_col':
                df_datetime = df_NAN.loc[:, df_NAN.columns[0]]
                df_NAN = df_NAN.drop(df_NAN.index[drop_rows])
                df_datetime = df_datetime.drop(df_datetime.index[drop_rows])
                df_NAN.drop(df_NAN.columns[[0]], axis=1, inplace=True)
                df_datetime = pd.to_datetime(df_datetime, dayfirst=dayfirst)
                
            elif datetime_mode == 'multi_col':
                df_datetime = df_NAN[df_NAN.columns[[0, 1, 2, 3, 4]]]

                df_NAN.drop(df_NAN.columns[[0, 1, 2, 3, 4]], axis=1, inplace=True)
                df_NAN = df_NAN.drop(drop_rows)
                df_datetime = df_datetime.drop(drop_rows)
                df_datetime = pd.to_datetime(df_datetime, dayfirst=dayfirst)

            else:
                print("please choose from 'single_col' or 'multi_col'")

            df_NAN = df_NAN.astype(float)
            meta = [[], []]
        else:
            print('   please choose "APGMer","MetOcean" or None as DataBase mode')
            return None

        df_NAN.index = df_datetime
        df = df_NAN.dropna(how='all')
        meta[0].append("NANs")
        meta[1].append(len(df_NAN) - len(df))
        meta[0].insert(0, "DataSet")
        meta[1].insert(0, file)
        NAMES.append(meta[0])
        VALUES.append(meta[1])

        df.to_sql('Hindcast_raw_' + file_name, conn)

        df_resample = df.resample(resample_rate).mean()

        data_resampled.append(df_resample)

    df_ges = join_dataframes_by_index(data_resampled)

    df_ges.to_sql('Hindcast_combined', conn)

    # Metadata
    df_Meta = pd.DataFrame(columns=NAMES[0][1:], index=[values[0] for values in VALUES])

    for values in VALUES:
        df_Meta.loc[values[0], :] = values[1:]

    df_Meta.to_sql('MetaData', conn)

    conn.close()

    return db_path


def extract_data(db_file, table_name, column_names=None):
    """
    Load specified columns or all columns from a table in an SQLite database and return as a pandas DataFrame,
    if tables index is specified, it is set as the index of the dataframe

    Parameters:
    - db_file (str): Path to the SQLite database file.
    - table_name (str): Name of the table to query.
    - column_names (list of str, optional): List of column names to retrieve. If None, retrieves all columns.

    Returns:
    - pandas.DataFrame: DataFrame containing the query results, including the row index as a column.

    Raises:
    - ValueError: If the table or columns do not exist.
    """
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if not cursor.fetchone():
            raise ValueError(f"Table '{table_name}' does not exist in the database.")

        if column_names:
            # Check if specified columns exist in the table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns_info = cursor.fetchall()
            table_columns = [info[1] for info in columns_info]
            for col in column_names:
                if col not in table_columns:
                    raise ValueError(f"Column '{col}' does not exist in table '{table_name}'.")
            # Enclose column names with special characters in double quotes
            cols = ', '.join([f'"{col}"' for col in column_names])
        else:
            # If no columns are specified, select all columns
            cols = '*'

        # Add ROWID to the columns to retrieve
        query = f"SELECT {cols} FROM {table_name};"
        df = pd.read_sql_query(query, conn)

        # set index of sql database to index of dataframe
        cursor.execute(f"PRAGMA index_list('{table_name}');")
        indexes = cursor.fetchall()
        cursor.execute(f"PRAGMA index_info('{indexes[0][1]}');")

        index_info = cursor.fetchall()

        if len(index_info) > 0:
            index_col_name = index_info[0][2]

            cursor.execute(f'SELECT "{index_col_name}" FROM "{table_name}";')
            index_col = cursor.fetchall()

            index_col = [info[0] for info in index_col]

            df.index = index_col

        return df

    except Error as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        if conn:
            conn.close()


def RWI(Hs, Tp, f_0, gamma_mode='default'):
    """calculates the RWI, using the JONSWAP sepectrum for Hs/Tp lists, arrays or Series"""
    RWI_list = []
    for Hs_curr, Tp_curr in zip(Hs, Tp):
        if Tp_curr != 0:
            S = gl.JONSWAP(f_0, Tp_curr, Hs_curr, gamma_mode=gamma_mode)[0]
        else:
            S = float('nan')
        RWI_list.append(np.sqrt(S))
    return RWI_list


def WaveBreak_steep(Hs, Tp, d, steep_crit):
    """"calculates, if the Wave (or seastate) is likley to breaking, eavaluated by wave steepness steep_crit

    Parameters:
        Hs: List, np array or Series, Wave Heigt
        Hs: List, np array or Series, Wave Period
        steep_crit: float, steepness criteria for wave steepness

    Returns:
        break_steep_bool_list: list, list of bools, True if Wave Breaks
        lamda_list: list, list of wavelengths, calculated with dispersion equation
    """

    break_steep_bool_list = []
    lamda_list = []
    steepness_list = []
    for HS, TP in zip(Hs, Tp):

        if TP != 0:
            omega = 1 / TP * 2 * np.pi

            k = gl.k_aus_omega(omega, d)
            lamda = 2 * np.pi / k
            steepness = HS / lamda
            break_steep_bool = steepness > steep_crit

        else:
            lamda = float('nan')
            break_steep_bool = float('nan')
            steepness = float('nan')
        break_steep_bool_list.append(break_steep_bool)
        steepness_list.append(steepness)
        lamda_list.append(lamda)

    return break_steep_bool_list, lamda_list, steepness_list


def angles(mode, N, start, **kwargs):
    width = kwargs.get("width", None)

    ang = []
    ang_mod = []

    if mode == 'full':
        width = 360 / N

    for i, _ in enumerate(range(N)):
        ang.append([start, start + width])
        ang_mod.append([start % 360, (start + width) % 360])
        start = start + width

    return ang, ang_mod


def ExtremeValues(x, N_itter=1000, intervall_mode='percentile', intervall_algorithm='1', T_Return_single=None, time_window_offset=0.5, perc_up=95, perc_down=5, freq_samp=1):
    if T_Return_single is None:
        T_Return_single = [1, 10, 40, 50]

    # sample maxiamal values
    x_max, years_newyear, years_start = gl.x_max_sampeling(x, time_window_offset)

    # sort values by size
    x_sorted_series = x_max.sort_values(ascending=True)
    x_sortet = np.array(x_sorted_series)
    N = len(x_sortet)

    # calculate gumbel coeffs of real data
    beta, mu = gl.Gumbel_coeff(x_max)

    # grid theorie (gumbel)
    x_grid = np.linspace(-(max(x_max) - min(x_max)) * 0.2 + min(x_max),
                         (max(x_max) - min(x_max)) * 0.2 + max(x_max), 100)

    F_theroie = gl.gumbel_P(x_grid, beta, mu)
    f_theroie = gl.gumbel_p(x_grid, beta, mu)

    # real
    F_real = [(n + 0.5) / N for n in range(N)]

    # theoretische erwartungswerte nach gumbelverteilung
    x_theorie = gl.gumbel_P_inv(F_real, beta, mu)

    # confidence intervall über x_theorie
    band_down_xth, _, band_up_xth, _ = gl.Intervall_conf_1(len(x_sortet), N_itter, beta, mu, perc_up, perc_down)

    T_R_x_real = [1 / ((1 - F_curr) * freq_samp) if F_curr != 1 else float("nan") for F_curr in F_real]
    # confidence intervall über TP
    band_down, middle, band_up, T_R_grid = gl.gumbel_conf_intervall(x_max, beta=beta, mu=mu, algorithm=intervall_algorithm, mode=intervall_mode, N_itter=N_itter,
                                                                    T_max=T_Return_single[-1])

    # single Returnperiods
    single_middle = [np.interp(temp_T, T_R_grid, middle) for temp_T in T_Return_single]
    single_down = [np.interp(temp_T, T_R_grid, band_down) for temp_T in T_Return_single]
    single_up = [np.interp(temp_T, T_R_grid, band_up) for temp_T in T_Return_single]

    points = pd.DataFrame({
        "x_max": x_sortet,
        "x_theorie": x_theorie,
        "F_real": F_real,
        "band_down": band_down_xth,
        "band_up": band_up_xth,
        "T_R_x_max": T_R_x_real})

    points.index = x_sorted_series.index

    grid = pd.DataFrame({
        "F_theorie": F_theroie,
        "f_theorie": f_theroie,
        "x_grid": x_grid})

    T_return = pd.DataFrame({"band_down": band_down,
                             "middle": middle,
                             "band_up": band_up,
                             "T_R_grid": T_R_grid})

    T_return_single = pd.DataFrame({
        "T_Return": T_Return_single,
        "middle": single_middle,
        "up": single_up,
        "down": single_down})

    out = {"points": points,
           "grid": grid,
           "years": {"newyear": years_newyear,
                     "start_window": years_start},
           "T_return_single": T_return_single,
           "T_return": T_return,
           "meta": {"N_itter": N_itter,
                    "intervall_mode": intervall_mode,
                    "intervall_algorithm": intervall_algorithm,
                    "perc_up": perc_up,
                    "perc_down": perc_down,
                    "freq_samp": freq_samp}}

    return out


# def write_DEL_database(db_path, DEL):

# alt
def Weibull_fit(Data_Sec, Input):
    global INFO_LOG

    def Weibull_fit(x):
        bin_size, center, count = gl.calculate_histogram(x)

        Weibull, params = gl.fit_weibull_distribution(x, center)

        Weibull = pd.Series(Weibull, index=center)

        return {'curve': Weibull, 'params': params, 'hist_count': count, 'hist_bin_size': bin_size}

    WEIBULL = {}

    INFO_LOG += "Weibull Parameters  \n"

    for key, dict_curr in Data_Sec.items():
        x = dict_curr[Input["col_name_values"]]
        WEIBULL[key] = Weibull_fit(x)

        INFO_LOG += f"{key}: " + "\n"
        INFO_LOG += gl.write_dict(WEIBULL[key]["params"])

    return WEIBULL


def DEl_Condensed(v_m, H_s, T_p, gamma, count, proj_path, exe_path):
    Table = pd.DataFrame(index=v_m)
    Added = pd.DataFrame()

    if gamma is None:
        gamma = 3.3

    isnan = np.isnan(H_s)

    proj_name = os.path.basename(proj_path)

    DEL_raw = gl.calc_JBOOST(exe_path, proj_name, H_s[~isnan], T_p[~isnan], gamma)
    Meta_data = gl.read_lua_values(proj_path, ["design_life", "N_ref", "SN_slope"])

    skal_time = (Meta_data["design_life"] * 365.25 * 24)

    for col in DEL_raw.columns:

        vm_vise = []
        weighted = []

        for DEl_curr, count_curr in zip(DEL_raw[col], count[~isnan]):
            # calculation DEL from count
            DEL_weigted = (DEl_curr ** Meta_data["SN_slope"] * Meta_data["N_ref"]) / skal_time * count_curr

            # including N_ref and SN-slope
            DEL_normalised = (DEL_weigted / Meta_data["N_ref"]) ** (1 / Meta_data["SN_slope"])

            vm_vise.append(DEL_normalised)
            weighted.append(DEL_weigted)

        Table[col] = 0
        Table[col] = Table[col].astype(float)
        Table.loc[isnan.values == False, col] = vm_vise
        Added[col] = pd.Series((np.nansum(weighted) / Meta_data["N_ref"]) ** (
                1 / Meta_data["SN_slope"]))

    Table["count"] = count
    Table.index = v_m

    return Table, Added


def DEL_points(DEL, v_m, v_m_edges, design_life, N_ref, SN_slope):
    Table = pd.DataFrame()
    Added = pd.DataFrame()

    count, _, bin_id = sc.stats.binned_statistic(v_m, v_m, statistic='count', bins=v_m_edges)
    skal_time = (design_life * 365.25 * 24)

    for col in DEL.columns:

        DEL_curr = DEL[col]

        vm_vise = []

        for n_v_bin in range(1, len(v_m_edges)):
            DEL_curr_vm_sec = DEL_curr.loc[bin_id == n_v_bin]

            # calculation DEL from count
            temp = (DEL_curr_vm_sec ** SN_slope * N_ref) / skal_time

            # including N_ref and SN-slope
            DEL_normalised = (np.nansum(temp) / N_ref) ** (1 / SN_slope)

            vm_vise.append(DEL_normalised)

        added = (np.nansum((DEL_curr.values ** SN_slope * N_ref) / skal_time) / N_ref) ** (
                1 / SN_slope)

        Table[col] = vm_vise
        Added[col] = pd.Series(added)

    Table["count"] = count

    v_m_edges = np.array(v_m_edges)
    v_m_mids = (v_m_edges[1:] + v_m_edges[:-1]) / 2

    Table.index = v_m_mids

    return Table, Added


def histogramm(x, x_max=None, x_min=None, auto_size=True, bin_size=None):
    bin_size_soll = bin_size

    if x_min is None:
        x_min = np.min(x)

    if x_max is None:
        x_max = np.max(x)

    # get significant Digits
    x_str = [str(value) for value in x]
    lenths = [len(dig.split('.')[1]) for dig in x_str]
    sig_dig = max(lenths)
    if sig_dig > 3: sig_dig = 3

    # get minmal distance
    x_values = np.sort(x)

    # x_max = np.max(x_values)
    x_unique = np.unique(np.round(x_values, sig_dig))
    bin_size = round(min(x_unique[1:] - x_unique[:-1]), sig_dig)
    N_bins = int(np.ceil(x_max / bin_size))
    x_max = N_bins * bin_size

    if x_min < 0:
        N_bins_down = int(np.ceil(-x_min / bin_size))
        x_min = -N_bins_down * bin_size
        N_bins = N_bins + N_bins_down
    if x_min > 0:
        N_bins_down = int(np.floor(-x_min / bin_size))
        x_min = N_bins_down * bin_size
        N_bins = N_bins - N_bins_down
    if N_bins > 10000:
        bin_size = np.round(x_max / 1000, 3)
        N_bins = int(x_max / bin_size)

    # choose, if to supplied binsize is to small:

    if auto_size is False:
        if bin_size_soll is None:
            print("   if auto_size is False, bin_size has to be definded")
        else:
            bin_size = bin_size_soll
            N_bins = int(x_max / bin_size)

    edges = np.linspace(x_min - bin_size / 2, x_max + bin_size / 2, N_bins + 2)
    edges = np.round(edges, 4)
    count = [len(x_values[(x_values > edges[i]) & (x_values <= edges[i + 1])]) for i in range(len(edges) - 1)]
    center = (edges[1:] + edges[:-1]) / 2

    return bin_size, center, count


def weibull_fit(x):
    bin_size, center, count = gl.calculate_histogram(x)

    Weibull, params = gl.fit_weibull_distribution(x, center)

    weibull = pd.Series(Weibull, index=center)

    prob = np.array(count) / ((sum(count)) * bin_size)

    return bin_size, center, prob, weibull, params


# %% macro functions
def calc_VMHS(Vm, Hs, angle, angle_grid, colidents=None, **kwargs):
    """retruns list of VMHS segment objects for all segments in angle_grid, if angle_grid is None, omnidirectional is returned

    Arguments:
        Vm: Series, Wind-Speed data, index is stored in Segment.indize Object to link used data, Datetime format recomended, Series Name is saved in Segment.colnames['x']
        Hs: Series, Wave-Height data, same index as Vm required, Series Name is saved in Segment.colnames['y']
        angle: Series, Wave-Height data, same index as Vm required, Series Name is saved in Segment.angle_name
        angle_grid: List of List (,2) with angle pairs discribing the edges of the segments, if None, omnidirectional is returned

    optional:
        N_grid: int, default: 100
        weight_y: bool, default: False
        deg_reg: int, default: 3
        model_reg: str, int, default: 'poly'
        cut_reg: int, default: 0
        weighting_reg: int, default: 0
        zone_reg: list, default: [None,None]
        zone_line: list, default: [None,None]
        bin_min: int, default: 0

    return:
        Data_Out: list of segment objects
    """

    N_grid = kwargs.get('N_grid', 100)
    deg_reg = kwargs.get('deg_reg', 3)
    model_reg = kwargs.get('model_reg', 'poly')
    cut_reg = kwargs.get('cut_reg', 0)
    weighting_reg = kwargs.get('weighting_reg', 0)
    zone_reg = kwargs.get('zone_reg', [None, None])
    zone_line = kwargs.get('zone_line', [None, None])
    bin_min = kwargs.get('bin_min', 0)
    perc_mean = kwargs.get('perc_mean', 50)
    avrg_method = kwargs.get('avrg_method', 'mean')
    make_monotone = kwargs.get('make_monotone', 'mean')

    Data_Out = []
    grid = np.linspace(0, max(Vm), N_grid + 1)

    if angle_grid is None:
        # omni
        df = pd.concat([Vm, Hs], axis=1)
        VMHS_DATA = condensation(df[Vm.name], df[Hs.name], grid,
                                 N_grid=N_grid,
                                 deg_reg=deg_reg,
                                 model_reg=model_reg,
                                 cut_reg=cut_reg,
                                 weighting_reg=weighting_reg,
                                 zone_reg=zone_reg,
                                 zone_line=zone_line,
                                 bin_min=bin_min,
                                 perc_mean=perc_mean,
                                 avrg_method=avrg_method,
                                 make_monotone=make_monotone)

        temp = Segment(0, angles=None,
                       result=VMHS_DATA,
                       colnames={'x': Vm.name, 'y': Hs.name},
                       indizes=list(df.index),
                       angle_name=None)

        Data_Out.append(temp)

    else:
        num = 1
        # Grid festlegen

        for angle_segment in angle_grid:
            df = pd.concat([Vm, Hs, angle], axis=1)
            df_filt = gl.filter_dataframe(df, angle.name, angle_segment[0], angle_segment[1])

            VMHS_DATA = condensation(df_filt[Vm.name], df_filt[Hs.name], grid,
                                     deg_reg=deg_reg,
                                     model_reg=model_reg,
                                     cut_reg=cut_reg,
                                     weighting_reg=weighting_reg,
                                     zone_reg=zone_reg,
                                     zone_line=zone_line,
                                     bin_min=bin_min,
                                     perc_mean=perc_mean,
                                     avrg_method=avrg_method,
                                     make_monotone=make_monotone)

            temp = Segment(num, angles=[angle_segment[0], angle_segment[1]], indizes=list(df_filt.index), result=VMHS_DATA,
                           colnames={'x': Vm.name, 'y': Hs.name}, angle_name=angle.name)
            Data_Out.append(temp)
            num = num + 1

    return Data_Out


def calc_HSTP(Hs, Tp, angle, angle_grid, **kwargs):
    """retruns list of HSTP segment objects for all segments in angle_grid, if angle_grid is None, omnidirectional is returned

    Arguments:
        Hs: Series, Wave-Height data, index is stored in Segment.indize Object to link used data, Datetime format recomended, Series Name is saved in Segment.colnames['x']
        Tp: Series, Wave-Period data, same index as Vm required, Series Name is saved in Segment.colnames['y']
        angle: Series, Wave-Height data, same index as Vm required, Series Name is saved in Segment.angle_name
        angle_grid: List of List (,2) with angle pairs discribing the edges of the segments, if None, omnidirectional is returned

    optional:
        N_grid: int, default: 100
        weight_y: bool, default: False
        deg_reg: int, default: 3
        model_reg: str, int, default: 'poly'
        cut_reg: int, default: 0
        weighting_reg: int, default: 0
        zone_reg: list, default: [None,None]
        zone_line: list, default: [None,None]
        bin_min: int, default: 0

    return:
        Data_Out: list of segment objects
    """

    N_grid = kwargs.get('N_grid', 100)
    deg_reg = kwargs.get('deg_reg', 3)
    model_reg = kwargs.get('model_reg', 'poly')
    cut_reg = kwargs.get('cut_reg', 0)
    weighting_reg = kwargs.get('weighting_reg', 0)
    zone_reg = kwargs.get('zone_reg', [None, None])
    zone_line = kwargs.get('zone_line', [None, None])
    bin_min = kwargs.get('bin_min', 0)
    quantile = kwargs.get('quantile', False)
    quant_up = kwargs.get('quant_up', None)
    quant_low = kwargs.get('quant_low', None)
    perc = kwargs.get('percentiles', [33, 66])
    perc_mean = kwargs.get('perc_mean', 50)
    avrg_method = kwargs.get('avrg_method', 'mean')
    make_monotone = kwargs.get('make_monotone', 'mean')

    Data_Out = []

    # Grid festlegen
    grid = np.linspace(0, max(Hs), N_grid + 1)

    if angle_grid is None:
        # omni
        Table_cond = condensation(Hs, Tp, grid,
                                  deg_reg=deg_reg,
                                  model_reg=model_reg,
                                  cut_reg=cut_reg,
                                  weighting_reg=weighting_reg,
                                  zone_reg=zone_reg,
                                  zone_line=zone_line,
                                  bin_min=bin_min,
                                  percentiles=perc,
                                  perc_mean=perc_mean,
                                  avrg_method=avrg_method,
                                  make_monotone=make_monotone
                                  )

        if quantile:

            if (len(perc) > 1) & (quant_up is not None) & (quant_low is not None):
                # todo: fehlernachricht
                key_low = f'{perc[0]}th percentile result plot'
                key_up = f'{perc[-1]}th percentile result plot'

                Table_cond["quantile"] = quantiles(Table_cond[key_low], Table_cond["mean result plot"], Table_cond[key_up], quant_low, quant_up)

        temp = Segment(0, angles=None,
                       result=Table_cond,
                       angle_name=None,
                       colnames={'x': Hs.name, 'y': Tp.name},
                       indizes=list(Hs.index))

        Data_Out.append(temp)

    else:
        df = pd.concat([Hs, Tp, angle], axis=1)

        for num, angle_segment in enumerate(angle_grid):

            df_filt = gl.filter_dataframe(df, angle.name, angle_segment[0], angle_segment[1])

            Table_cond = condensation(df_filt[Hs.name], df_filt[Tp.name], grid,
                                      N_grid=N_grid,
                                      deg_reg=deg_reg,
                                      model_reg=model_reg,
                                      cut_reg=cut_reg,
                                      weighting_reg=weighting_reg,
                                      zone_reg=zone_reg,
                                      zone_line=zone_line,
                                      bin_min=bin_min,
                                      percentiles=perc,
                                      perc_mean=perc_mean,
                                      avrg_method=avrg_method,
                                      make_monotone=make_monotone
                                      )

            if quantile:

                if (len(perc) > 1) & (quant_up is not None) & (quant_low is not None):
                    # todo: fehlernachricht
                    key_low = f'{perc[0]}th percentile result plot'
                    key_up = f'{perc[-1]}th percentile result plot'

                    Table_cond["quantile"] = quantiles(Table_cond[key_low], Table_cond["mean result plot"], Table_cond[key_up], quant_low, quant_up)

            temp = Segment(num=num, angles=[angle_segment[0], angle_segment[1]], result=Table_cond, indizes=list(df_filt.index), angle_name=angle.name,
                           colnames={'x': Hs.name, 'y': Tp.name})

            Data_Out.append(temp)

    return Data_Out


def calc_VMTP(vmhs, hstp, fill_value_interp=False):
    """takes output from VMHS_calc und HSTP_calc (list of angle Sgements) and cross-correlates the results to get the VMTP correlation
    - Takes angle information from vmhs, assumes vmhs and hstp in fitting order of angle-segments!!
    - needs Segment.result object to be a pd.Dataframe with "mean result plot" and "x" in vmhs result dataframes and "x" and "mean result plot" or "quantile" in hstp dataframes ("quantile" overwrites "mean result plot")!)
    - indizes (from basedata) needs to be intialized for counts!
    """

    def get_group_labels(series):
        group_labels = (series != series.shift()).cumsum() * series
        grouped = series.groupby(group_labels)
        return grouped.groups[2], grouped.groups[4]

    VMTP = []
    num = 0

    for vmhs_curr, hstp_curr in zip(vmhs, hstp):

        vmhs_curr_data = vmhs_curr.result
        hstp_curr_data = hstp_curr.result
        vmtp_curr_data = pd.DataFrame()

        if "quantile" in hstp_curr_data.keys():
            key_tp = "quantile"
        else:
            key_tp = "mean result plot"

        HS = vmhs_curr_data['mean result plot']
        mask = ~np.isnan(HS)
        nans = np.empty(len(HS))
        nans[:] = np.nan

        VM_grid = pd.Series(nans)

        VM_grid[mask] = vmhs_curr_data.loc[mask, 'x']

        HS_grid = hstp_curr_data['x']

        VM_res = gl.interpolate_increasing_decreasing(HS_grid, HS[mask], VM_grid[mask])

        if fill_value_interp:

            vm_left_beg = VM_grid[mask].iloc[0]
            vm_right_end = VM_grid[mask].iloc[-1]
            vm_left_end = VM_res[~np.isnan(VM_res)][0]
            vm_right_beg = VM_res[~np.isnan(VM_res)][-1]

            Tp_res = hstp_curr_data.loc[:, key_tp].copy()
            bool_const = ~np.isnan(Tp_res) & np.isnan(VM_res)

            indx_lin_left, indx_lin_right = get_group_labels(bool_const)

            Tp_res[indx_lin_left] = Tp_res.loc[~np.isnan(VM_res)].iloc[0]
            Tp_res[indx_lin_right] = Tp_res.loc[~np.isnan(VM_res)].iloc[-1]

            VM_res[indx_lin_left] = np.linspace(vm_left_beg, vm_left_end, len(indx_lin_left))
            VM_res[indx_lin_right] = np.linspace(vm_right_beg, vm_right_end, len(indx_lin_right))

        else:
            Tp_res = hstp_curr_data.loc[~np.isnan(VM_res), key_tp].copy()

        vmtp_curr_data['x'] = VM_res

        vmtp_curr_data.loc[~np.isnan(VM_res), 'mean result plot'] = hstp_curr_data.loc[~np.isnan(VM_res), key_tp]

        col_name_vm = vmhs_curr.colnames['x']
        col_name_hs = hstp_curr.colnames['y']
        angle = vmhs_curr.angle_name
        angles = vmhs_curr.angles

       # look, if resulting VM_res was sucsessfull, when not, no correlation from VMHS can be found due to constant correlation
        if len(np.where(~np.isnan(VM_res))[0]) < 3:
            print(
                f"   no correlation between Vm and Hs is found for section {vmhs_curr.angles}, Hs assumed constant over Vm (check if its the case, otherwise faulty result possible)")
            VM_res[mask] = VM_grid[mask]
            Hs_konst = np.mean(HS[mask])
            Tp_const = np.interp(Hs_konst, hstp_curr_data['x'], hstp_curr_data[key_tp])
            Tp_res = pd.Series(nans)
            Tp_res.iloc[mask == True] = Tp_const

        vmtp_curr_data['x'] = VM_res
        vmtp_curr_data.loc[~np.isnan(
            VM_res), 'mean plot'] = Tp_res

        vmtp_curr = Segment(num, result=vmtp_curr_data, colnames={'x': col_name_vm, 'y': col_name_hs}, angle_name=angle, angles=angles, indizes=vmhs_curr.indizes)

        VMTP.append(vmtp_curr)
        num = num + 1

    return VMTP


def calc_Roseplot(angle, magnitude, angle_segments):
    r_edges = gl.auto_ticks(0, max(magnitude), fix_end=False, edges=False)
    r_middle = (r_edges[1:] - r_edges[:-1]) / 2
    counts = pd.DataFrame(index=r_middle)
    for angle_segment in angle_segments:
        df = pd.concat([angle, magnitude], axis=1)
        df_filt = gl.filter_dataframe(df, angle.name, angle_segment[0], angle_segment[1])

        count, _, _ = sc.stats.binned_statistic(df_filt[magnitude.name], df_filt[magnitude.name], statistic='count', bins=r_edges)

        angle_midpoint = gl.angle_midpoints([angle_segment[0]], [angle_segment[1]])

        counts[str(angle_midpoint[0])] = pd.Series(count, index=counts.index)

    return counts, r_edges


def calc_RWI(Hs, Tp, angle, angle_grid, f_0):
    """retruns list of RWI segment objects for all segments in angle_grid, if angle_grid is None, omnidirectional is returned

    Arguments:
        Hs: Series, Wave-Height data, index is stored in Segment.indize Object to link used data, Datetime format recomended, Series Name is saved in Segment.colnames['x']
        Tp: Series, Wave-Period data, same index as Vm required, Series Name is saved in Segment.colnames['y']
        angle: Series, Wave-Height data, same index as Vm required, Series Name is saved in Segment.angle_name
        angle_grid: List of List (,2) with angle pairs discribing the edges of the segments, if None, omnidirectional is returned
        f_0: first resonance freq

    return:
        Data_Out: list of segment objects
        RWI:info:  {'RWI_max': RWI_max, 'HS_max': HS_max, 'TP_max': TP_max}
    """
    Data_Out = []
    HS_max = 0
    TP_max = 0
    RWI_max = 0

    if angle_grid is None:

        RWI_list = RWI(Hs, Tp, f_0)

        RWI_df = pd.DataFrame(RWI_list, index=Hs.index)

        temp = Segment(0, angles=None, result=RWI_df, colnames={'x': Hs.name, 'y': Tp.name}, angle_name=None, indizes=list(Hs.index))
        Data_Out.append(temp)

        RWI_max = max(RWI_list)
        HS_max = Hs[RWI_list == RWI_max]
        TP_max = Tp[RWI_list == RWI_max]

    else:

        df = pd.concat([Hs, Tp, angle], axis=1)

        for num, angle_segment in enumerate(angle_grid):

            df_filt = gl.filter_dataframe(df, angle.name, angle_segment[0], angle_segment[1])

            RWI_list = RWI(df_filt[Hs.name], df_filt[Tp.name], f_0)

            RWI_df = pd.DataFrame(RWI_list, index=df_filt[Hs.name].index)

            temp = Segment(num, angles=[angle_segment[0], angle_segment[1]], result=RWI_df, colnames={'x': Hs.name, 'y': Tp.name}, angle_name=angle.name,
                           indizes=list(df_filt[Hs.name].index))
            Data_Out.append(temp)

            if max(RWI_list) > RWI_max:
                RWI_max = max(RWI_list)
                HS_max = df_filt[Hs.name][RWI_list == RWI_max]
                TP_max = df_filt[Tp.name][RWI_list == RWI_max]

    return Data_Out, {'RWI_max': RWI_max, 'HS_max': HS_max, 'TP_max': TP_max}


def calc_tables(vmhs, vm_grid, vm_data):
    VMHS = vmhs
    #    VMTP = hstp

    VMHS_OUT = []
    VMTP_OUT = []

    vm_step = vm_grid[0] - vm_grid[1]
    vm_center = (vm_grid[1:] + vm_grid[:-1]) / 2

    Hsdata = pd.DataFrame()

    Tpdata = pd.DataFrame()
    isDATA_VMHS = pd.DataFrame()
    isDATA_VMTP = pd.DataFrame()
    Count = pd.DataFrame()

    Hsdata['v_m'] = [
        f"{vm_grid[i]} to < {vm_grid[i + 1]}" for i in range(len(vm_grid) - 1)]
    Tpdata['v_m'] = [
        f"{vm_grid[i]} to < {vm_grid[i + 1]}" for i in range(len(vm_grid) - 1)]
    isDATA_VMHS['v_m'] = [
        f"{vm_grid[i]} to < {vm_grid[i + 1]}" for i in range(len(vm_grid) - 1)]
    isDATA_VMTP['v_m'] = [
        f"{vm_grid[i]} to < {vm_grid[i + 1]}" for i in range(len(vm_grid) - 1)]
    Count['v_m'] = [
        f"{vm_grid[i]} to < {vm_grid[i + 1]}" for i in range(len(vm_grid) - 1)]

    # VMHS
    i = 0
    for vmhs_curr in VMHS:
        vmhs_result = vmhs_curr.result

        # VM and HS data from Regression curves
        HS_VMHS = vmhs_result["mean result plot"]
        VM_VMHS = vmhs_result["x"]

        # get VM grid and count infomation for Table
        index_vmhs_curr = vmhs_curr.indizes
        vm_curr = vm_data[vm_data.index.isin(pd.to_datetime(index_vmhs_curr))]

        count_table, x_edges, _ = sc.stats.binned_statistic(
            vm_curr, vm_curr, statistic='count', bins=vm_grid)

        # 'isdata' for table where count != 0
        Vm_table_center = (x_edges[:-1] + x_edges[1:]) / 2
        isdata = count_table != 0
        count_table = np.array(count_table)

        # set span between fist and last isdata==True to True to catch values in between
        spanned_isdata_table = np.empty(len(Hsdata['v_m']), dtype=bool)
        spanned_isdata_table[:] = False
        spanned_isdata_table[min(
            np.where(isdata)[0]):max(np.where(isdata)[-1] + 1)] = True

        HS = np.empty(len(Hsdata['v_m']))
        HS[:] = np.nan

        # interplate in span for the non nan values of the regression data
        HS[spanned_isdata_table] = sc.interpolate.interp1d(
            VM_VMHS[~np.isnan(HS_VMHS)], HS_VMHS[~np.isnan(HS_VMHS)], fill_value='extrapolate')(
            Vm_table_center[spanned_isdata_table])

        # vm_zone_VMHS = [Vm_table_center[spanned_isdata_table].iloc[0] -
        #                vm_step / 2, Vm_table_center[spanned_isdata_table].iloc[-1] + vm_step / 2]

        result = pd.DataFrame()

        result["vm"] = vm_center
        result["vm_edges"] = [[vm_grid[i], vm_grid[i + 1]] for i in range(len(vm_grid) - 1)]
        result["value"] = HS
        result["isdata"] = isdata
        result["count"] = count_table

        temp = Segment(i, angles=vmhs_curr.angles, result=result, colnames=vmhs_curr.colnames, indizes=index_vmhs_curr, angle_name=vmhs_curr.angle_name)
        VMHS_OUT.append(temp)
        i = i + 1

    # VMTP
    i = 0

    # for vmtp_curr in VMTP:
    #
    #     vmtp_result = vmtp_curr.result
    #
    #     # VM and TP data from cross correlation, VM has no leading nans and starts late, not gridded,
    #     # TP gridded and starts with leading nans
    #
    #     TP_VMTP = VMTP[ck]["grid"]["mean plot"]
    #     VM_VMTP = VMTP[ck]["grid"]["x"]
    #
    #     # get VM grid and count infomation for Table
    #     df = gl.grid_pointcloud_in_x(
    #         DATA_SEC[ck][COLNAMES["Vm"]], DATA_SEC[ck][COLNAMES["Hs"]], vm_grid)
    #
    #     # 'isdata' for table where count != 0
    #     count_table = df["count"]
    #     Vm_table_center = df["x"]
    #     isdata = count_table != 0
    #
    #     # 'isdata' for table where there is no data because of no cross correlation
    #     bool_corr = (Vm_table_center > min(VM_VMTP[~np.isnan(VM_VMTP)]) - vm_step / 2) & (
    #             Vm_table_center < (max(VM_VMTP[~np.isnan(VM_VMTP)]) + vm_step / 2))
    #     isdata_table = isdata & bool_corr
    #
    #     # set span between fist and last isdata==True to True to catch values in between
    #
    #     spanned_isdata_table = np.empty(len(Hsdata['v_m']), dtype=bool)
    #     spanned_isdata_table[:] = False
    #     spanned_isdata_table[min(
    #         np.where(isdata_table)[0]):max(np.where(isdata_table)[-1] + 1)] = True
    #
    #     TP = np.empty(len(Hsdata['v_m']))
    #     TP[:] = np.nan
    #     TP[spanned_isdata_table] = sc.interpolate.interp1d(
    #         VM_VMTP[~np.isnan(TP_VMTP)], TP_VMTP[~np.isnan(TP_VMTP)], fill_value='extrapolate')(
    #         Vm_table_center.iloc[spanned_isdata_table])
    #
    #     vm_zone_VMTP = [Vm_table_center[spanned_isdata_table].iloc[0] -
    #                     vm_step / 2, Vm_table_center[spanned_isdata_table].iloc[-1] + vm_step / 2]
    #
    #     VMZone_VMTP[ck] = vm_zone_VMTP
    #
    #     Tpdata[ck] = TP
    #     isDATA_VMTP[ck] = isdata_table
    #
    #     i = i + 1
    #
    # OUT = {"VMHS": {}}
    # OUT["VMHS"]["table_content"] = Hsdata
    # OUT["VMHS"]["VM_Zone"] = VMZone_VMHS
    # OUT["VMHS"]["isData"] = isDATA_VMHS
    #
    # OUT["VMTP"] = {}
    # OUT["VMTP"]["table_content"] = Tpdata
    # OUT["VMTP"]["VM_Zone"] = VMZone_VMTP
    # OUT["VMTP"]["isData"] = isDATA_VMTP
    #
    # OUT["count"] = Count
    # OUT["Vm_edges"] = Vm_table_edges

    return VMHS_OUT


def calc_WaveBreak_Steep(Hs, Tp, angle, angle_grid, steep_crit, d):
    """retruns list of WaveBreakSteep segment objects for all segments in angle_grid, if angle_grid is None, omnidirectional is returned

    Arguments:
        Hs: Series, Wave-Height data, index is stored in Segment.indize Object to link used data, Datetime format recomended, Series Name is saved in Segment.colnames['x']
        Tp: Series, Wave-Period data, same index as Vm required, Series Name is saved in Segment.colnames['y']
        angle: Series, Wave-Height data, same index as Vm required, Series Name is saved in Segment.angle_name
        angle_grid: List of List (,2) with angle pairs discribing the edges of the segments, if None, omnidirectional is returned
        steep_crit: speeness criteria
        d: water depth

    return:
        Data_Out: list of segment objects
    """

    Data_Out = []

    if angle_grid is None:

        break_steep_bool_list, lamda_list, steepness = WaveBreak_steep(Hs, Tp, d, steep_crit)
        break_steep_bool = pd.Series(break_steep_bool_list, name='bool_break')
        lamda = pd.Series(lamda_list, name='lamda')
        steepness = pd.Series(steepness, name='steepness')

        result = pd.concat([break_steep_bool, lamda, steepness, pd.Series(data=Hs.values, name=Hs.name)], axis=1, ignore_index=False)

        temp = Segment(0, angles=None, result=result, colnames={'x': Hs.name, 'y': Tp.name}, angle_name=None, indizes=list(Hs.index))
        Data_Out.append(temp)

    else:
        df = pd.concat([Hs, Tp, angle], axis=1)

        for num, angle_segment in enumerate(angle_grid):
            df_filt = gl.filter_dataframe(df, angle.name, angle_segment[0], angle_segment[1])

            break_steep_bool_list, lamda_list, steepness = WaveBreak_steep(df_filt[Hs.name], df_filt[Tp.name], d, steep_crit)
            break_steep_bool = pd.Series(break_steep_bool_list, name='bool_break')
            lamda = pd.Series(lamda_list, name='lamda')
            steepness = pd.Series(steepness, name='steepness')

            result = pd.concat([break_steep_bool, lamda, steepness], axis=1)

            temp = Segment(num, angles=[angle_segment[0], angle_segment[1]], result=result, colnames={'x': Hs.name, 'y': Tp.name}, angle_name=angle.name,
                           indizes=list(df_filt.index))
            Data_Out.append(temp)

    return Data_Out


def calc_angle_deviation_tables(angle_orig, angle_comp, v_m, angle_grid, **kwargs):
    """creates missalinment tables for every angle section of angle_orig stored in angle_grid. 
    The rows are definded by a histogram of the v_m vektor. 
    The columns are named after the missalinment of the compare angle angle_comp and is calcultate with: diff =  angle_com - angle_orig"""

    v_m_zone = kwargs.get('v_m_zone', None)
    v_m_step = kwargs.get('v_m_step', 1)
    N_angle_comp_sec = kwargs.get('N_angle_comp_sec', 12)
    Data_Out = []
    indexes = angle_orig.index

    # v_m zone generation
    if v_m_zone is None:
        v_m_zone = [0, max(v_m.values)]

    else:
        if v_m_zone[1] is None:
            v_m_zone[1] = max(v_m.values)

    if np.isnan(v_m_zone[1]):
        vm_zone[1] = max(DATA[COLNAMES["Vm"]])

    v_m_range = gl.range_stepfix(v_m_step, v_m_zone)

    comp_ang, comp_ang_mod = angles('full', N_angle_comp_sec, -15)

    df = pd.DataFrame({"angle_orig": angle_orig.values, "angle_comp": angle_comp.values, "v_m": v_m}, index=indexes)

    for num, angle_segment in enumerate(angle_grid):

        df_filt = gl.filter_dataframe(df, "angle_orig", angle_segment[0], angle_segment[1])
        midpoint_orig = gl.angle_midpoints([angle_segment[0]], [angle_segment[1]])[0]
        AngleDev = pd.DataFrame()
        AngleDev.index = (v_m_range[1:] + v_m_range[:-1]) / 2
        missalignments = []
        midpoints_comp = []

        for comp_ang_mod_curr in comp_ang_mod:

            midpoint_comp = gl.angle_midpoints([comp_ang_mod_curr[0]], [comp_ang_mod_curr[1]])[0]

            missalignment = (midpoint_comp - midpoint_orig) % 360

            missalignments.append(missalignment)
            midpoints_comp.append(midpoint_comp)

            df_filt_filt = gl.filter_dataframe(df_filt, "angle_comp", comp_ang_mod_curr[0], comp_ang_mod_curr[1])

            if len(df_filt_filt) != 0:
                Count, _, _ = sc.stats.binned_statistic(df_filt_filt["v_m"].values, df_filt_filt["v_m"].values, statistic='count', bins=v_m_range)
            else:
                Count = np.zeros(len(v_m_range) - 1)

            Count = Count / len(df_filt)

            AngleDev[midpoint_comp] = Count

        # sorting

        _, count_cols = zip(*sorted(zip(missalignments, midpoints_comp)))
        count_cols = count_cols
        AngleDev = AngleDev.reindex(columns=count_cols)

        missalinment_rename = {midpoints_comp[i]: missalignments[i] for i in range(len(missalignments))}

        AngleDev = AngleDev.rename(columns=missalinment_rename)

        temp = Segment(num,
                       angles=[angle_segment[0], angle_segment[1]],
                       result=AngleDev,
                       colnames={'ang_orig': angle_orig.name, 'ang_comp': angle_comp.name, 'v_m': v_m.name},
                       angle_name=angle_orig.name,
                       indizes=list(df_filt.index))

        Data_Out.append(temp)

    return Data_Out


def calc_ExtemeValues(x, angles, angle_grid, T_return_single=None, conf_inter_mode=None, conf_inter_algorithm=None, N_itter=None, freq_samp=None, perc_up=None, perc_down=None,
                      time_window_offset=None):
    Data_Out = []

    if angle_grid is None:
        result = ExtremeValues(x,
                               intervall_mode=conf_inter_mode,
                               intervall_algorithm=conf_inter_algorithm,
                               T_Return_single=T_return_single,
                               time_window_offset=time_window_offset,
                               perc_up=perc_up,
                               perc_down=perc_down,
                               freq_samp=freq_samp,
                               N_itter=N_itter)

        temp = Segment(0, angles=None, result=result, colnames={'x': x.name, 'angle': angles.name}, angle_name=None, indizes=list(x.index))
        Data_Out.append(temp)

    else:
        df = pd.concat([x, angles], axis=1)
        for num, angle_segment in enumerate(angle_grid):
            df_filt = gl.filter_dataframe(df, angles.name, angle_segment[0], angle_segment[1])
            result = ExtremeValues(df_filt[x.name],
                                   intervall_mode=conf_inter_mode,
                                   intervall_algorithm=conf_inter_algorithm,
                                   T_Return_single=T_return_single,
                                   time_window_offset=time_window_offset,
                                   perc_up=perc_up,
                                   perc_down=perc_down,
                                   freq_samp=freq_samp,
                                   N_itter=N_itter)

            temp = Segment(0, angles=angle_segment, result=result, colnames={'x': x.name, 'angle': angles.name}, angle_name=angles.name, indizes=list(df_filt.index))
            Data_Out.append(temp)
    return Data_Out


def update_DEL_db(db_path, Hs, Tp, gamma, proj_path=None, input_path=None, exe_path=None):
    """function, that handels updating DEL values from a Database or calculates the missing information using JBOOST

        tries to calculate missing information using JBOOST, but also compares the Metadata provided in proj_path and input_path for no unnessesary calculations
        """

    if os.path.normpath(proj_path) == os.path.normpath(exe_path):
        print('   invalid JBOOST procject file location, do no place inside ./JBOOST')
        return None
    else:
        JBOOST_proj_path_new = shutil.copy(proj_path, exe_path)
    proj_name = os.path.basename(proj_path)

    if os.path.normpath(input_path) == os.path.normpath(exe_path):
        print('   invalid JBOOST procject input file location, do no place inside ./JBOOST')
        return None
    else:
        shutil.copy(input_path, exe_path)

    Var_lua = gl.read_lua_values(JBOOST_proj_path_new, ['seabed_level', 'design_life', 'N_ref', 'SN_slope', 'res_Nodes'])

    Meta_Curr = {"d": -Var_lua["seabed_level"],
                 "design_life": Var_lua["design_life"],
                 "N_ref": Var_lua["N_ref"],
                 "SN_slope": Var_lua["SN_slope"],
                 "Hs": Hs.name,
                 "Tp": Tp.name,
                 "gamma": gamma.name,
                 'input': input_path}

    print(f'   used variables from {proj_path}:')
    print(f"   {gl.write_dict(Var_lua)}")

    table_name = gl.check_meta_in_valid_db(db_path, Meta_Curr)

    if len(table_name) != 0:
        table_name = table_name[0]
        print('   data with same input parameters found, checking for nodes and timeframe')
        DEL_base_sql = gl.export_df_from_sql(db_path, table_name)

        node_coles = DEL_base_sql.columns
        columnnames_data = list(node_coles)

        # node difference
        nodes = []
        for col in node_coles:
            node = ''.join([s for s in col if s.isdigit()])
            node = float(node)
            nodes.append(node)

        nodes_db = np.unique(nodes)

        nodes_proj = Var_lua["res_Nodes"][1:-1].split(',')
        nodes_proj = [float(node) for node in nodes_proj]

        nodes_proj = set(nodes_proj)
        nodes_db = set(nodes_db)

        new_nodes = nodes_proj.difference(nodes_db)
        both_nodes = nodes_proj.intersection(nodes_db)
        if len(new_nodes) > 0:
            nodes_to_add = True
        else:
            nodes_to_add = False

        # find data, which covers all nans of the nodes provided in nodes_proj
        SQL_temp = gl.filter_df_cols_by_keywords(DEL_base_sql, [str(int(node)) for node in nodes_proj])
        SQL_temp = SQL_temp.dropna()

        # timeframe difference
        in_df = Hs.index

        in_db = SQL_temp.index
        df2_in_df1 = Hs.index.isin(in_db)
        in_df_not_in_db = Hs.index[~df2_in_df1]

        if len(in_df_not_in_db) > 0:
            time_to_add = True
        else:
            time_to_add = False

        if time_to_add and not nodes_to_add:
            print(f'   New time found in run (N = {len(in_df_not_in_db)}), but no new nodes. Calculating missing timepoints to add to db {table_name}, this might take a long time')
            # write database data, which will not be calculated in db frame

            # wirte calculated data, which was not in database in db frame
            DEL_temp = gl.calc_JBOOST(exe_path, proj_name, Hs.loc[in_df_not_in_db], Tp.loc[in_df_not_in_db], gamma.loc[in_df_not_in_db])

        if not time_to_add and nodes_to_add:
            print(f'   New nodes found in run ({new_nodes}), but no new timestep. Calculating the new nodes, this might take a long time')

            gl.write_lua_variables(JBOOST_proj_path_new, {'res_Nodes': new_nodes})

            # write calculated dataframe where it is calculated in db frame
            DEL_temp = gl.calc_JBOOST(exe_path, proj_name, Hs, Tp, gamma)

        if time_to_add and nodes_to_add:
            print(f'   New nodes found in run ({new_nodes}).')
            print(f'   New time found in run (N = {len(in_df_not_in_db)}). Calculating nodes and timepoints, this might take a long time')

            gl.write_lua_variables(JBOOST_proj_path_new, {'res_Nodes': new_nodes})
            # calculate all new nodes and timepoints in dataframe and write in db_frame
            DEL_temp_01 = gl.calc_JBOOST(exe_path, proj_name, Hs.loc[in_df], Tp.loc[in_df], gamma.loc[in_df])

            # calculate the new data for the other nodes
            gl.write_lua_variables(JBOOST_proj_path_new, {'res_Nodes': both_nodes})
            DEL_temp_02 = gl.calc_JBOOST(exe_path, proj_name, Hs.loc[in_df_not_in_db], Tp.loc[in_df_not_in_db], gamma.loc[in_df])

            DEL_temp = gl.merge_dataframes(DEL_temp_02, DEL_temp_01)

        if time_to_add or nodes_to_add:

            #  wait_statement = input("overwrite data in database with current calculation? \n write y for yes and n for no: ")
            wait_statement = 'y'
            if wait_statement == 'y':
                DEL_save = gl.merge_dataframes(DEL_temp, DEL_base_sql)

                write_db = True
            else:
                write_db = False
                DEL_save = None

        else:
            print('   database with same input parameters, same nodes and same or wider timeframe found, no calculation needed')
            write_db = False

    else:
        print(f"   crate new database table and calculate all {len(Hs)} datapoints for all nodes, this might take a long time")
        DEL_save = gl.calc_JBOOST('./JBOOST/', proj_name, Hs, Tp, gamma)

        write_db = True
        columnnames_data = list(DEL_save.columns)

    if write_db:
        table_name = gl.write_DEL_base(db_path, DEL_save, Meta_Curr)

    return table_name, columnnames_data


def calc_Validation(df_DEL, Vm, angle, vmhs_table, vmtp_table, proj_path, input_path, exe_path):
    """retruns list of Validation segment objects for all segments in angle_grid, if angle_grid is None, omnidirectional is returned

    Arguments:
        Vm: Series, Wind-Speed data, index is stored in Segment.indize Object to link used data, Datetime format recomended, Series Name is saved in Segment.colnames['x']
        angle: Series, Wave-Height data, same index as Vm required, Series Name is saved in Segment.angle_name

    optional:

    return:
        Data_Out: list of segment objects
    """
    # extract Nodes from DEL data
    node_coles = df_DEL.columns
    Data_Out = []
    # node difference
    nodes = []
    for col in node_coles:
        node = ''.join([s for s in col if s.isdigit()])
        node = float(node)
        nodes.append(node)

    nodes_db = set(nodes)

    if os.path.normpath(proj_path) == os.path.normpath(exe_path):
        print('   invalid JBOOST procject file location, do no place inside ./JBOOST')
        return None
    else:
        JBOOST_proj_path_new = shutil.copy(proj_path, exe_path)

    if os.path.normpath(input_path) == os.path.normpath(exe_path):
        print('   invalid JBOOST procject input file location, do no place inside ./JBOOST')
        return None
    else:
        shutil.copy(input_path, exe_path)

    gl.write_lua_variables(JBOOST_proj_path_new, {'res_Nodes': nodes_db})

    Meta_data = gl.read_lua_values(JBOOST_proj_path_new, ["design_life", "N_ref", "SN_slope"])

    num = 0
    for vmhs_curr, vmtp_curr in zip(vmhs_table, vmtp_table):

        table_condensed, added_condensed = DEl_Condensed(vmtp_curr.result["vm"],
                                                         vmhs_curr.result["value"],
                                                         vmtp_curr.result["value"],
                                                         None,
                                                         vmhs_curr.result["count"],
                                                         JBOOST_proj_path_new,
                                                         exe_path)

        angle_segment = vmhs_curr.angles

        if angle_segment is not None:
            id_filt = gl.filter_dataframe(angle, [angle.name], [angle_segment[0]], [angle_segment[1]]).index
            angle_name = angle.name
        else:
            id_filt = angle.index
            angle_name = None

        v_m_edges = [vm_curr[0] for vm_curr in vmhs_curr.result["vm_edges"].values]
        v_m_edges.append(vmhs_curr.result["vm_edges"].values[-1][1])

        table_points, added_points = DEL_points(df_DEL.loc[id_filt], Vm.loc[id_filt], v_m_edges, Meta_data["design_life"], Meta_data["N_ref"], Meta_data["SN_slope"])

        out = {"condensed": {"vm_vise": table_condensed,
                             "added": added_condensed},
               "hindcast": {"vm_vise": table_points,
                            "added": added_points},
               "meta": Meta_data}

        V_m_name = vmhs_curr.colnames["x"]
        H_s_name = vmhs_curr.colnames["y"]
        T_p_name = vmtp_curr.colnames["y"]

        temp = Segment(num, angles=angle_segment, indizes=list(id_filt), result=out, angle_name=angle_name,
                       colnames={'Hindcast': {'v_m': V_m_name, 'H_s': H_s_name, 'T_p': T_p_name},
                                 'DELs': list(table_points.columns)})

        Data_Out.append(temp)
        num = num + 1

    return Data_Out


def calc_histogram(x, angle, angle_grid):
    Data_Out = []

    if angle_grid is None:
        # omni
        bin_size, center, count = histogramm(x)
        histo_data = {"bin_size": bin_size, "center": center, "count": count}

        temp = Segment(0, angles=None,
                       result=histo_data,
                       angle_name=None,
                       colnames={'x': x.name},
                       indizes=list(x.index))

        Data_Out.append(temp)

    else:
        num = 1
        # Grid festlegen

        for angle_segment in angle_grid:
            df = pd.concat([x, angle], axis=1)
            df_filt = gl.filter_dataframe(df, angle.name, angle_segment[0], angle_segment[1])

            bin_size, center, count = histogramm(df_filt[x.name])

            histo_data = {"bin_size": bin_size, "center": center, "count": count}

            temp = Segment(num, angles=[angle_segment[0], angle_segment[1]], indizes=list(df_filt.index), result=histo_data, angle_name=angle.name,
                           colnames={'x': x.name})
            Data_Out.append(temp)
            num = num + 1

    return Data_Out


def calc_weibull(x, angle, angle_grid):
    Data_Out = []

    if angle_grid is None:
        # omni
        bin_size, center, prob, weibull, weibull_params = weibull_fit(x)
        histo_data = {"bin_size": bin_size,
                      "center": center,
                      "prob": prob,
                      "weibull": weibull,
                      "weibull_params": weibull_params
                      }

        temp = Segment(0, angles=None,
                       result=histo_data,
                       angle_name=None,
                       colnames={'x': x.name},
                       indizes=list(x.index))

        Data_Out.append(temp)

    else:
        num = 1
        # Grid festlegen

        for angle_segment in angle_grid:
            df = pd.concat([x, angle], axis=1)
            df_filt = gl.filter_dataframe(df, angle.name, angle_segment[0], angle_segment[1])

            bin_size, center, prob, weibull, weibull_params = weibull_fit(df_filt[x.name])
            histo_data = {"bin_size": bin_size,
                          "center": center,
                          "prob": prob,
                          "weibull": weibull,
                          "weibull_params": weibull_params
                          }

            temp = Segment(num, angles=[angle_segment[0], angle_segment[1]], indizes=list(df_filt.index), result=histo_data, angle_name=angle.name,
                           colnames={'x': x.name})
            Data_Out.append(temp)
            num = num + 1

    return Data_Out
# %% DataBaseHandling


# load data f
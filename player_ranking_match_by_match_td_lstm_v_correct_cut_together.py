import ast
import unicodedata
import csv
import os
import scipy.io as sio
from scipy.stats.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


class GoalImpactMetric:

    def __init__(self):
        self.FEATURE_TYPE = 5
        self.calibration = True
        self.ITERATE_NUM = 30
        self.MODEL_TYPE = "v4"
        self.BATCH_SIZE = 32
        self.learning_rate = 1e-4
        self.pre_initialize = False
        self.MAX_TRACE_LENGTH = 2
        self.if_correct_velocity = "_v_correct_"
        self.ROUND_NUMBER = 60

        self.IS_POSSIBILITY = True
        self.IS_DIFFERENCE = True
        if self.IS_DIFFERENCE:
            self.DIFFERENCE_TYPE = "back_difference_"

        self.PLAYER_ID_DICT_ALL_BY_MATCH = {}
        self.PLAYER_ID_DICT_ALL = {}
        self.PLAYER_INTEREST = ['G', 'A', 'P', 'PlayerName', 'GP', 'PlusMinus', 'PIM', 'PointPerGame', 'PPG', 'PPP',
                                'SHG', 'SHP',
                                'GWG', 'OTG', 'S', 'ShootingPercentage', 'TOIPerGame', 'ShiftsPerGame',
                                'FaceoffWinPercentage']

        if self.learning_rate == 1e-6:
            self.learning_rate_write = 6
        elif self.learning_rate == 1e-5:
            self.learning_rate_write = 5
        elif self.learning_rate == 1e-4:
            self.learning_rate_write = 4
        else:
            raise ValueError("undefined learning_rate")

        if self.pre_initialize:
            self.pre_initialize_save = "_pre_initialize"
        else:
            self.pre_initialize_save = ""

        self.model_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hybrid-RNN-Hockey-Training-All-feature{0}-scale" \
                                    "-neg_reward{1}_length-dynamic".format(str(self.FEATURE_TYPE),
                                                                           self.if_correct_velocity)

        # self.model_data_store_dir = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Training-All-feature{0}-scale-neg_reward{1}".format(str(self.FEATURE_TYPE), self.if_correct_velocity)

        self.data_path = "/cs/oschulte/Galen/Hockey-data-entire/Hockey-Match-All-data"
        self.dir_all = os.listdir(self.data_path)

        self.player_info_dir = "../player_statistic/players_2015_2016.csv"

        self.skater_info_dir = "../player_statistic/skater_stats_2015_2016_original.csv"

        self.data_name = "model_three_cut_together_predict_Feature{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}{6}".format(
            str(self.FEATURE_TYPE),
            str(self.ITERATE_NUM),
            str(self.learning_rate_write),
            str(self.BATCH_SIZE),
            str(self.MAX_TRACE_LENGTH),
            str(self.MODEL_TYPE),
            self.if_correct_velocity)

    # state_data_name = "model_state_cut_together_predict_Fea{0}_Iter{1}_lr{2}_Batch{3}_MaxLength{4}_Type{5}".format(
    #     str(FEATURE_TYPE), str(ITERATE_NUM), str(6), str(8), str(MAX_TRACE_LENGTH), MODEL_TYPE)

    def aggregate_values(self):
        """
        aggregate value for each player
        """
        for calibration_dir_game in os.listdir(self.model_data_store_dir):
            # model_state_data_name = state_model_data_store_dir + "/" + calibration_dir_game + "/" + state_data_name + ".mat"
            # model_state_data = (sio.loadmat(model_state_data_name))[state_data_name]
            for file_name in os.listdir(self.model_data_store_dir + "/" + calibration_dir_game):
                if file_name == self.data_name + ".mat":
                    model_data_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                    model_data = (sio.loadmat(model_data_name))[self.data_name]
                elif file_name.startswith("playerId"):
                    playerIds_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                    playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
                elif file_name.startswith("home_identifier"):
                    home_identifier_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                    home_identifier = (sio.loadmat(home_identifier_name))["home_identifier"][0]
                else:
                    continue

            for player_Index in range(0, len(playerIds)):
                playerId = playerIds[player_Index]
                model_value = model_data[player_Index]
                # model_state_value = model_state_data[player_Index]
                ishome = home_identifier[player_Index]
                player_value = self.PLAYER_ID_DICT_ALL.get(playerId)
                if player_value is None:
                    if ishome:
                        if self.IS_POSSIBILITY:
                            self.PLAYER_ID_DICT_ALL.update(
                                {playerId: {
                                    "value all": (model_value[0] - model_value[1]) / (
                                            model_value[0] + model_value[1] + abs(model_value[2]))}})
                            # "state value": (model_state_value[0] - model_state_value[1]) / (
                            # model_state_value[0] + model_state_value[1])}})
                        else:
                            self.PLAYER_ID_DICT_ALL.update({playerId: {"value all": model_value[0] - model_value[1]}})
                            # "state value": model_state_value[0] - model_state_value[1]}})
                    else:
                        if self.IS_POSSIBILITY:
                            self.PLAYER_ID_DICT_ALL.update(
                                {playerId: {
                                    "value all": (model_value[1] - model_value[0]) / (
                                            model_value[0] + model_value[1] + abs(model_value[2]))}})
                            # "state value": (model_state_value[1] - model_state_value[0]) / (
                            # model_state_value[0] + model_state_value[1])}})
                        else:
                            self.PLAYER_ID_DICT_ALL.update({playerId: {"value all": model_value[1] - model_value[0]}})
                            # "state value": model_state_value[1] - model_state_value[0]}})
                else:
                    if ishome:
                        if self.IS_POSSIBILITY:
                            player_value_number = player_value.get("value all") + (model_value[0] - model_value[1]) / (
                                    model_value[0] + model_value[1] + abs(model_value[2]))
                            # player_state_value_number = player_value.get("state value") + (model_state_value[0] - model_state_value[1])/(model_state_value[0] + model_state_value[1])
                        else:
                            player_value_number = player_value.get("value all") + model_value[0] - model_value[1]
                            # player_state_value_number = player_value.get("state value") + model_state_value[0] - \
                            #                             model_state_value[1]
                    else:
                        if self.IS_POSSIBILITY:
                            player_value_number = player_value.get("value all") + (model_value[1] - model_value[0]) / (
                                    model_value[0] + model_value[1] + abs(model_value[2]))
                            # player_state_value_number = player_value.get("state value") + (model_state_value[1] - model_state_value[0])/(model_state_value[0] + model_state_value[1])
                        else:
                            player_value_number = player_value.get("value all") + model_value[1] - model_value[0]
                            # player_state_value_number = player_value.get("state value") + model_state_value[1] - \
                            # model_state_value[0]
                    self.PLAYER_ID_DICT_ALL.update(
                        {playerId: {"value all": player_value_number}})
                    # {playerId: {"value": player_value_number, "state value": player_state_value_number}}), "state value": player_state_value_number}})
                    # break

    def aggregate_diff_values(self):
        """
        aggregate value for each player
        """
        for calibration_dir_game in os.listdir(self.model_data_store_dir):
            # model_state_data_name = state_model_data_store_dir + "/" + calibration_dir_game + "/" + state_data_name + ".mat"
            # model_state_data = (sio.loadmat(model_state_data_name))[state_data_name]
            for file_name in os.listdir(self.model_data_store_dir + "/" + calibration_dir_game):
                if file_name == self.data_name + ".mat":
                    model_data_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                    model_data = (sio.loadmat(model_data_name))[self.data_name]
                elif file_name.startswith("playerId"):
                    playerIds_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                    playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
                elif file_name.startswith("home_identifier"):
                    home_identifier_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                    home_identifier = (sio.loadmat(home_identifier_name))["home_identifier"][0]
                elif "training_data_dict_all_name" in file_name:
                    training_data_dict_all_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                    training_data_dict_all = ((sio.loadmat(training_data_dict_all_name))["training_data_dict_all_name"])
                else:
                    continue

            for player_Index in range(0, len(playerIds)):
                playerId = playerIds[player_Index]
                model_value = model_data[player_Index]

                if player_Index - 1 >= 0:
                    training_data_dict_all_pre = training_data_dict_all[player_Index - 1]
                    training_data_dict_all_pre_str = unicodedata.normalize('NFKD', training_data_dict_all_pre).encode(
                        'ascii', 'ignore')
                    training_data_dict_all_pre_dict = ast.literal_eval(training_data_dict_all_pre_str)

                    if training_data_dict_all_pre_dict.get('action') == "goal":
                        model_value_pre = model_data[player_Index]
                    else:
                        model_value_pre = model_data[player_Index - 1]
                else:
                    model_value_pre = model_data[player_Index]

                if player_Index + 1 < len(playerIds):
                    training_data_dict_all_nex = training_data_dict_all[player_Index]
                    training_data_dict_all_nex_str = unicodedata.normalize('NFKD', training_data_dict_all_nex).encode(
                        'ascii', 'ignore')
                    training_data_dict_all_nex_dict = ast.literal_eval(training_data_dict_all_nex_str)

                    if training_data_dict_all_nex_dict.get('action') == "goal":
                        model_value_nex = model_data[player_Index]
                    else:
                        model_value_nex = model_data[player_Index + 1]

                else:
                    model_value_nex = model_data[player_Index]

                if model_value[2] < 0:
                    model_value[2] = 0
                if model_value_pre[2] < 0:
                    model_value_pre[2] = 0
                if model_value_nex[2] < 0:
                    model_value_nex[2] = 0

                ishome = home_identifier[player_Index]
                player_value = self.PLAYER_ID_DICT_ALL.get(playerId)
                if player_value is None:
                    if ishome:
                        if self.IS_POSSIBILITY:
                            home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            end_model_value = abs(model_value[2]) / (
                                        model_value[0] + model_value[1] + abs(model_value[2]))

                            home_model_value_pre = model_value_pre[0] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            away_model_value_pre = model_value_pre[1] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            end_model_value_pre = abs(model_value_pre[2]) / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                            home_model_value_nex = model_value_nex[0] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            away_model_value_nex = model_value_nex[1] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            end_model_value_nex = abs(model_value_nex[2]) / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (home_model_value - home_model_value_pre) - (
                                        away_model_value - away_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (home_model_value_nex - home_model_value) - (
                                        away_model_value_nex - away_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (home_model_value_nex - home_model_value_pre) - (
                                        away_model_value_nex - away_model_value_pre)

                            self.PLAYER_ID_DICT_ALL.update(
                                {playerId: {
                                    "value all": q_value}})
                            # "state value": (model_state_value[0] - model_state_value[1]) / (
                            # model_state_value[0] + model_state_value[1])}})
                        else:
                            home_model_value = model_value[0]
                            away_model_value = model_value[1]
                            end_model_value = abs(model_value[2])
                            home_model_value_pre = model_value_pre[0]
                            away_model_value_pre = model_value_pre[1]
                            end_model_value_pre = abs(model_value_pre[2])
                            home_model_value_nex = model_value_nex[0]
                            away_model_value_nex = model_value_nex[1]
                            end_model_value_nex = abs(model_value_nex[2])

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (home_model_value - home_model_value_pre) - (
                                        away_model_value - away_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (home_model_value_nex - home_model_value) - (
                                        away_model_value_nex - away_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (home_model_value_nex - home_model_value_pre) - (
                                        away_model_value_nex - away_model_value_pre)

                            self.PLAYER_ID_DICT_ALL.update({playerId: {"value all": q_value}})
                            # "state value": model_state_value[0] - model_state_value[1]}})
                    else:
                        if self.IS_POSSIBILITY:
                            home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            end_model_value = abs(model_value[2]) / (
                                        model_value[0] + model_value[1] + abs(model_value[2]))

                            home_model_value_pre = model_value_pre[0] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            away_model_value_pre = model_value_pre[1] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            end_model_value_pre = abs(model_value_pre[2]) / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                            home_model_value_nex = model_value_nex[0] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            away_model_value_nex = model_value_nex[1] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            end_model_value_nex = abs(model_value_nex[2]) / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (away_model_value - away_model_value_pre) - (
                                        home_model_value - home_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (away_model_value_nex - away_model_value) - (
                                        home_model_value_nex - home_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (away_model_value_nex - away_model_value_pre) - (
                                        home_model_value_nex - home_model_value_pre
                                )

                            self.PLAYER_ID_DICT_ALL.update(
                                {playerId: {
                                    "value all": q_value}})
                        else:
                            home_model_value = model_value[0]
                            away_model_value = model_value[1]
                            end_model_value = abs(model_value[2])
                            home_model_value_pre = model_value_pre[0]
                            away_model_value_pre = model_value_pre[1]
                            end_model_value_pre = abs(model_value_pre[2])
                            home_model_value_nex = model_value_nex[0]
                            away_model_value_nex = model_value_nex[1]
                            end_model_value_nex = abs(model_value_nex[2])

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (away_model_value - away_model_value_pre) - (
                                        home_model_value - home_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (away_model_value_nex - away_model_value) - (
                                        home_model_value_nex - home_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (away_model_value_nex - away_model_value_pre) - (
                                        home_model_value_nex - home_model_value_pre
                                )

                            self.PLAYER_ID_DICT_ALL.update(
                                {playerId: {
                                    "value all": q_value}})
                else:
                    if ishome:
                        if self.IS_POSSIBILITY:
                            home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            end_model_value = abs(model_value[2]) / (
                                        model_value[0] + model_value[1] + abs(model_value[2]))

                            home_model_value_pre = model_value_pre[0] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            away_model_value_pre = model_value_pre[1] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            end_model_value_pre = abs(model_value_pre[2]) / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                            home_model_value_nex = model_value_nex[0] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            away_model_value_nex = model_value_nex[1] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            end_model_value_nex = abs(model_value_nex[2]) / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (home_model_value - home_model_value_pre) - (
                                        away_model_value - away_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (home_model_value_nex - home_model_value) - (
                                        away_model_value_nex - away_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (home_model_value_nex - home_model_value_pre) - (
                                        away_model_value_nex - away_model_value_pre)

                            player_value_number = player_value.get("value all") + q_value
                        else:
                            home_model_value = model_value[0]
                            away_model_value = model_value[1]
                            end_model_value = abs(model_value[2])
                            home_model_value_pre = model_value_pre[0]
                            away_model_value_pre = model_value_pre[1]
                            end_model_value_pre = abs(model_value_pre[2])
                            home_model_value_nex = model_value_nex[0]
                            away_model_value_nex = model_value_nex[1]
                            end_model_value_nex = abs(model_value_nex[2])

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (home_model_value - home_model_value_pre) - (
                                        away_model_value - away_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (home_model_value_nex - home_model_value) - (
                                        away_model_value_nex - away_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (home_model_value_nex - home_model_value_pre) - (
                                        away_model_value_nex - away_model_value_pre)

                            player_value_number = player_value.get("value all") + q_value

                    else:
                        if self.IS_POSSIBILITY:
                            home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            end_model_value = abs(model_value[2]) / (
                                        model_value[0] + model_value[1] + abs(model_value[2]))

                            home_model_value_pre = model_value_pre[0] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            away_model_value_pre = model_value_pre[1] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            end_model_value_pre = abs(model_value_pre[2]) / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                            home_model_value_nex = model_value_nex[0] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            away_model_value_nex = model_value_nex[1] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            end_model_value_nex = abs(model_value_nex[2]) / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (away_model_value - away_model_value_pre) - (
                                        home_model_value - home_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (away_model_value_nex - away_model_value) - (
                                        home_model_value_nex - home_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (away_model_value_nex - away_model_value_pre) - (
                                        home_model_value_nex - home_model_value_pre
                                )
                            player_value_number = player_value.get("value all") + q_value


                        else:
                            home_model_value = model_value[0]
                            away_model_value = model_value[1]
                            end_model_value = abs(model_value[2])
                            home_model_value_pre = model_value_pre[0]
                            away_model_value_pre = model_value_pre[1]
                            end_model_value_pre = abs(model_value_pre[2])
                            home_model_value_nex = model_value_nex[0]
                            away_model_value_nex = model_value_nex[1]
                            end_model_value_nex = abs(model_value_nex[2])

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (away_model_value - away_model_value_pre) - (
                                        home_model_value - home_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (away_model_value_nex - away_model_value) - (
                                        home_model_value_nex - home_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (away_model_value_nex - away_model_value_pre) - (
                                        home_model_value_nex - home_model_value_pre
                                )
                            player_value_number = player_value.get("value all") + q_value

                    self.PLAYER_ID_DICT_ALL.update(
                        {playerId: {"value all": player_value_number}})
                    # {playerId: {"value": player_value_number, "state value": player_state_value_number}}), "state value": player_state_value_number}})
                    # break

    def aggregate_match_values(self, game_target_dir, teamId_target):
        """

        :param game_target_dir:
        :param teamId_target:
        :return:
        """
        for file_name in os.listdir(self.model_data_store_dir + "/" + game_target_dir):
            if file_name == self.data_name + ".mat":
                model_data_name = self.model_data_store_dir + "/" + game_target_dir + "/" + file_name
                model_data = (sio.loadmat(model_data_name))[self.data_name]
            elif file_name.startswith("playerId"):
                playerIds_name = self.model_data_store_dir + "/" + game_target_dir + "/" + file_name
                playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
            elif file_name.startswith("teamId"):
                teamIds_name = self.model_data_store_dir + "/" + game_target_dir + "/" + file_name
                teamIds = (sio.loadmat(teamIds_name))["teamId"][0]
            elif file_name.startswith("home_identifier"):
                home_identifier_name = self.model_data_store_dir + "/" + game_target_dir + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_identifier"][0]
            else:
                continue

        for player_Index in range(0, len(playerIds)):
            playerId = playerIds[player_Index]
            teamId = teamIds[player_Index]
            if int(teamId_target) == int(teamId):
                model_value = model_data[player_Index]
                # model_state_value = model_state_data[player_Index]
                ishome = home_identifier[player_Index]
                player_value = self.PLAYER_ID_DICT_ALL_BY_MATCH.get(playerId)
                if player_value is None:
                    if ishome:
                        if self.IS_POSSIBILITY:
                            self.PLAYER_ID_DICT_ALL_BY_MATCH.update(
                                {playerId: {
                                    "value": (model_value[0] - model_value[1]) / (
                                            model_value[0] + model_value[1] + abs(model_value[2]))}})
                            # "state value": (model_state_value[0] - model_state_value[1]) / (
                            # model_state_value[0] + model_state_value[1])}})
                        else:
                            self.PLAYER_ID_DICT_ALL_BY_MATCH.update(
                                {playerId: {"value": model_value[0] - model_value[1]}})
                            # "state value": model_state_value[0] - model_state_value[1]}})
                    else:
                        if self.IS_POSSIBILITY:
                            self.PLAYER_ID_DICT_ALL_BY_MATCH.update(
                                {playerId: {
                                    "value": (model_value[1] - model_value[0]) / (
                                            model_value[0] + model_value[1] + abs(model_value[2]))}})
                            # "state value": (model_state_value[1] - model_state_value[0]) / (
                            # model_state_value[0] + model_state_value[1])}})
                        else:
                            self.PLAYER_ID_DICT_ALL_BY_MATCH.update(
                                {playerId: {"value": model_value[1] - model_value[0]}})
                            # "state value": model_state_value[1] - model_state_value[0]}})
                else:
                    if ishome:
                        if self.IS_POSSIBILITY:
                            player_value_number = player_value.get("value") + (model_value[0] - model_value[1]) / (
                                    model_value[0] + model_value[1] + abs(model_value[2]))
                            # player_state_value_number = player_value.get("state value") + (model_state_value[0] - model_state_value[1])/(model_state_value[0] + model_state_value[1])
                        else:
                            player_value_number = player_value.get("value") + model_value[0] - model_value[1]
                            # player_state_value_number = player_value.get("state value") + model_state_value[0] - \
                            #                             model_state_value[1]
                    else:
                        if self.IS_POSSIBILITY:
                            player_value_number = player_value.get("value") + (model_value[1] - model_value[0]) / (
                                    model_value[0] + model_value[1] + abs(model_value[2]))
                            # player_state_value_number = player_value.get("state value") + (model_state_value[1] - model_state_value[0])/(model_state_value[0] + model_state_value[1])
                        else:
                            player_value_number = player_value.get("value") + model_value[1] - model_value[0]
                            # player_state_value_number = player_value.get("state value") + model_state_value[1] - \
                            # model_state_value[0]
                        self.PLAYER_ID_DICT_ALL_BY_MATCH.update(
                            {playerId: {"value": player_value_number}})
                    # {playerId: {"value": player_value_number, "state value": player_state_value_number}}), "state value": player_state_value_number}})
                    # break

    def aggregate_match_diff_values(self, calibration_dir_game, teamId_target):
        for file_name in os.listdir(self.model_data_store_dir + "/" + calibration_dir_game):
            if file_name == self.data_name + ".mat":
                model_data_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                model_data = (sio.loadmat(model_data_name))[self.data_name]
            elif file_name.startswith("playerId"):
                playerIds_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                playerIds = (sio.loadmat(playerIds_name))["playerId"][0]
            elif file_name.startswith("teamId"):
                teamIds_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                teamIds = (sio.loadmat(teamIds_name))["teamId"][0]
            elif file_name.startswith("home_identifier"):
                home_identifier_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                home_identifier = (sio.loadmat(home_identifier_name))["home_identifier"][0]
            elif "training_data_dict_all_name" in file_name:
                training_data_dict_all_name = self.model_data_store_dir + "/" + calibration_dir_game + "/" + file_name
                training_data_dict_all = ((sio.loadmat(training_data_dict_all_name))["training_data_dict_all_name"])
            else:
                continue

        for player_Index in range(0, len(playerIds)):
            playerId = playerIds[player_Index]
            teamId = teamIds[player_Index]
            if int(teamId_target) == int(teamId):
                model_value = model_data[player_Index]
                if player_Index - 1 >= 0:
                    training_data_dict_all_pre = training_data_dict_all[player_Index - 1]
                    training_data_dict_all_pre_str = unicodedata.normalize('NFKD', training_data_dict_all_pre).encode(
                        'ascii', 'ignore')
                    training_data_dict_all_pre_dict = ast.literal_eval(training_data_dict_all_pre_str)

                    if training_data_dict_all_pre_dict.get('action') == "goal":
                        model_value_pre = model_data[player_Index]
                    else:
                        model_value_pre = model_data[player_Index - 1]
                else:
                    model_value_pre = model_data[player_Index]

                if player_Index + 1 <= len(playerIds):
                    training_data_dict_all_nex = training_data_dict_all[player_Index]
                    training_data_dict_all_nex_str = unicodedata.normalize('NFKD', training_data_dict_all_nex).encode(
                        'ascii', 'ignore')
                    training_data_dict_all_nex_dict = ast.literal_eval(training_data_dict_all_nex_str)

                    if training_data_dict_all_nex_dict.get('action') == "goal":
                        model_value_nex = model_data[player_Index]
                    else:
                        model_value_nex = model_data[player_Index - 1]

                else:
                    model_value_nex = model_data[player_Index]

                if model_value[2] < 0:
                    model_value[2] = 0
                if model_value_pre[2] < 0:
                    model_value_pre[2] = 0
                if model_value_nex[2] < 0:
                    model_value_nex[2] = 0

                ishome = home_identifier[player_Index]
                player_value = self.PLAYER_ID_DICT_ALL_BY_MATCH.get(playerId)
                if player_value is None:
                    if ishome:
                        if self.IS_POSSIBILITY:
                            home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            end_model_value = abs(model_value[2]) / (
                                        model_value[0] + model_value[1] + abs(model_value[2]))

                            home_model_value_pre = model_value_pre[0] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            away_model_value_pre = model_value_pre[1] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            end_model_value_pre = abs(model_value_pre[2]) / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                            home_model_value_nex = model_value_nex[0] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            away_model_value_nex = model_value_nex[1] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            end_model_value_nex = abs(model_value_nex[2]) / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (home_model_value - home_model_value_pre) - (
                                        away_model_value - away_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (home_model_value_nex - home_model_value) - (
                                        away_model_value_nex - away_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (home_model_value_nex - home_model_value_pre) - (
                                        away_model_value_nex - away_model_value_pre)

                            self.PLAYER_ID_DICT_ALL_BY_MATCH.update(
                                {playerId: {
                                    "value": q_value}})
                            # "state value": (model_state_value[0] - model_state_value[1]) / (
                            # model_state_value[0] + model_state_value[1])}})
                        else:
                            home_model_value = model_value[0]
                            away_model_value = model_value[1]
                            end_model_value = abs(model_value[2])
                            home_model_value_pre = model_value_pre[0]
                            away_model_value_pre = model_value_pre[1]
                            end_model_value_pre = abs(model_value_pre[2])
                            home_model_value_nex = model_value_nex[0]
                            away_model_value_nex = model_value_nex[1]
                            end_model_value_nex = abs(model_value_nex[2])

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (home_model_value - home_model_value_pre) - (
                                        away_model_value - away_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (home_model_value_nex - home_model_value) - (
                                        away_model_value_nex - away_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (home_model_value_nex - home_model_value_pre) - (
                                        away_model_value_nex - away_model_value_pre)

                            self.PLAYER_ID_DICT_ALL_BY_MATCH.update({playerId: {"value": q_value}})
                            # "state value": model_state_value[0] - model_state_value[1]}})
                    else:
                        if self.IS_POSSIBILITY:
                            home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            end_model_value = abs(model_value[2]) / (
                                        model_value[0] + model_value[1] + abs(model_value[2]))

                            home_model_value_pre = model_value_pre[0] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            away_model_value_pre = model_value_pre[1] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            end_model_value_pre = abs(model_value_pre[2]) / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                            home_model_value_nex = model_value_nex[0] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            away_model_value_nex = model_value_nex[1] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            end_model_value_nex = abs(model_value_nex[2]) / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (away_model_value - away_model_value_pre) - (
                                        home_model_value - home_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (away_model_value_nex - away_model_value) - (
                                        home_model_value_nex - home_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (away_model_value_nex - away_model_value_pre) - (
                                        home_model_value_nex - home_model_value_pre
                                )

                            self.PLAYER_ID_DICT_ALL_BY_MATCH.update(
                                {playerId: {
                                    "value": q_value}})
                        else:
                            home_model_value = model_value[0]
                            away_model_value = model_value[1]
                            end_model_value = abs(model_value[2])
                            home_model_value_pre = model_value_pre[0]
                            away_model_value_pre = model_value_pre[1]
                            end_model_value_pre = abs(model_value_pre[2])
                            home_model_value_nex = model_value_nex[0]
                            away_model_value_nex = model_value_nex[1]
                            end_model_value_nex = abs(model_value_nex[2])

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (away_model_value - away_model_value_pre) - (
                                        home_model_value - home_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (away_model_value_nex - away_model_value) - (
                                        home_model_value_nex - home_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (away_model_value_nex - away_model_value_pre) - (
                                        home_model_value_nex - home_model_value_pre
                                )

                            self.PLAYER_ID_DICT_ALL_BY_MATCH.update(
                                {playerId: {
                                    "value": q_value}})
                else:
                    if ishome:
                        if self.IS_POSSIBILITY:
                            home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            end_model_value = abs(model_value[2]) / (
                                        model_value[0] + model_value[1] + abs(model_value[2]))

                            home_model_value_pre = model_value_pre[0] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            away_model_value_pre = model_value_pre[1] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            end_model_value_pre = abs(model_value_pre[2]) / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                            home_model_value_nex = model_value_nex[0] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            away_model_value_nex = model_value_nex[1] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            end_model_value_nex = abs(model_value_nex[2]) / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (home_model_value - home_model_value_pre) - (
                                        away_model_value - away_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (home_model_value_nex - home_model_value) - (
                                        away_model_value_nex - away_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (home_model_value_nex - home_model_value_pre) - (
                                        away_model_value_nex - away_model_value_pre)

                            player_value_number = player_value.get("value") + q_value
                        else:
                            home_model_value = model_value[0]
                            away_model_value = model_value[1]
                            end_model_value = abs(model_value[2])
                            home_model_value_pre = model_value_pre[0]
                            away_model_value_pre = model_value_pre[1]
                            end_model_value_pre = abs(model_value_pre[2])
                            home_model_value_nex = model_value_nex[0]
                            away_model_value_nex = model_value_nex[1]
                            end_model_value_nex = abs(model_value_nex[2])

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (home_model_value - home_model_value_pre) - (
                                        away_model_value - away_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (home_model_value_nex - home_model_value) - (
                                        away_model_value_nex - away_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (home_model_value_nex - home_model_value_pre) - (
                                        away_model_value_nex - away_model_value_pre)

                            player_value_number = player_value.get("value") + q_value

                    else:
                        if self.IS_POSSIBILITY:
                            home_model_value = model_value[0] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            away_model_value = model_value[1] / (model_value[0] + model_value[1] + abs(model_value[2]))
                            end_model_value = abs(model_value[2]) / (
                                        model_value[0] + model_value[1] + abs(model_value[2]))

                            home_model_value_pre = model_value_pre[0] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            away_model_value_pre = model_value_pre[1] / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))
                            end_model_value_pre = abs(model_value_pre[2]) / (
                                    model_value_pre[0] + model_value_pre[1] + abs(model_value_pre[2]))

                            home_model_value_nex = model_value_nex[0] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            away_model_value_nex = model_value_nex[1] / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))
                            end_model_value_nex = abs(model_value_nex[2]) / (
                                    model_value_nex[0] + model_value_nex[1] + abs(model_value_nex[2]))

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (away_model_value - away_model_value_pre) - (
                                        home_model_value - home_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (away_model_value_nex - away_model_value) - (
                                        home_model_value_nex - home_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (away_model_value_nex - away_model_value_pre) - (
                                        home_model_value_nex - home_model_value_pre
                                )
                            player_value_number = player_value.get("value") + q_value


                        else:
                            home_model_value = model_value[0]
                            away_model_value = model_value[1]
                            end_model_value = abs(model_value[2])
                            home_model_value_pre = model_value_pre[0]
                            away_model_value_pre = model_value_pre[1]
                            end_model_value_pre = abs(model_value_pre[2])
                            home_model_value_nex = model_value_nex[0]
                            away_model_value_nex = model_value_nex[1]
                            end_model_value_nex = abs(model_value_nex[2])

                            if self.DIFFERENCE_TYPE == "back_difference_":
                                q_value = (away_model_value - away_model_value_pre) - (
                                        home_model_value - home_model_value_pre)
                            elif self.DIFFERENCE_TYPE == "front_difference_":
                                q_value = (away_model_value_nex - away_model_value) - (
                                        home_model_value_nex - home_model_value)
                            elif self.DIFFERENCE_TYPE == "skip_difference_":
                                q_value = (away_model_value_nex - away_model_value_pre) - (
                                        home_model_value_nex - home_model_value_pre
                                )
                            player_value_number = player_value.get("value") + q_value

                    self.PLAYER_ID_DICT_ALL_BY_MATCH.update(
                        {playerId: {"value": player_value_number}})
                    # {playerId: {"value": player_value_number, "state value": player_state_value_number}}), "state value": player_state_value_number}})
                    # break

    def transfer_save_format_by_match(self, playerId_skateInfo_dict):
        player_value_dict_list = []
        player_Ids = self.PLAYER_ID_DICT_ALL_BY_MATCH.keys()
        for player_index in range(0, len(player_Ids)):
            player_value_dict = {}
            player_Id = player_Ids[player_index]
            player_round_value = self.PLAYER_ID_DICT_ALL_BY_MATCH.get(player_Id)
            player_all_value = self.PLAYER_ID_DICT_ALL.get(player_Id)
            player_skateInfo = playerId_skateInfo_dict.get(player_Id)
            if player_skateInfo is not None:
                player_value_dict.update({"playerId": player_Ids[player_index]})
                player_value_dict.update(player_round_value)
                player_value_dict.update(player_all_value)
                player_value_dict.update(player_skateInfo)
                player_value_dict_list.append(player_value_dict)
        return player_value_dict_list

    def write_csv(self, csv_name, data_record):
        with open(csv_name, 'w') as csvfile:
            fieldnames = (data_record[0]).keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for record in data_record:
                writer.writerow(record)

    def read_players_info(self):
        player_Info = {}
        first_row_flag = True
        with open(self.player_info_dir) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if first_row_flag:
                    first_row_flag = False
                    continue
                else:
                    firstname = row[2]
                    lastname = row[1]
                    playerId = row[0]
                    player_Info.update({firstname + " " + lastname: playerId})
        return player_Info

    def combine_playerId_to_skate_info(self, player_Info):
        first_row_flag = True
        playerId_skateInfo_dict = {}
        row_num = 0
        with open(self.skater_info_dir) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=';')
            for row in read_csv:
                # row_num += 1
                if first_row_flag:
                    first_row_flag = False
                    item_list = row
                    continue
                else:
                    # print row
                    player_name = row[1]
                    season = row[5]
                    if season == "Playoffs":
                        continue
                    player_id = player_Info.get(player_name)
                    if player_id is not None:
                        row_num += 1
                        item_record = {}
                        for interest_item in self.PLAYER_INTEREST:
                            interest_value = row[item_list.index(interest_item)]
                            item_record.update({interest_item: interest_value})
                        temp = playerId_skateInfo_dict.get(int(player_id))
                        if temp is not None:
                            print(player_name)
                        playerId_skateInfo_dict.update({int(player_id): item_record})
                    else:
                        print(player_name)
        print(row_num)
        return playerId_skateInfo_dict

    def team_match_statistic(self):
        gameInfo_dir = "/cs/oschulte/Galen/Hockey-data-entire"

        gameInfo_load = sio.loadmat(gameInfo_dir + '/gamesInfo.mat')
        gamesInfo = gameInfo_load['gamesInfo']

        Team_name_dict = {}
        Team_shortcut_dict = {}

        for ff in range(len(gamesInfo[0])):  # fixed bug
            gamesInfoTemp = gamesInfo[0, ff]
            gamesInfoId = unicodedata.normalize('NFKD', gamesInfoTemp['id'][0][0][0]).encode('ascii', 'ignore')
            gamesInfoVis = (gamesInfoTemp['visitors'])[0, 0]
            gamesInfoHome = (gamesInfoTemp['home'])[0, 0]

            gamesInfoHomeName = unicodedata.normalize('NFKD', gamesInfoHome['name'][0][0][0]).encode('ascii', 'ignore')
            gamesInfoHomeShortCut = unicodedata.normalize('NFKD', gamesInfoHome['shorthand'][0][0][0]).encode('ascii',
                                                                                                              'ignore')
            gamesInfoHomeId = unicodedata.normalize('NFKD', gamesInfoHome['id'][0][0][0]).encode('ascii', 'ignore')

            gamesInfoVisName = unicodedata.normalize('NFKD', gamesInfoVis['name'][0][0][0]).encode('ascii', 'ignore')
            gamesInfoVisShortCut = unicodedata.normalize('NFKD', gamesInfoVis['shorthand'][0][0][0]).encode('ascii',
                                                                                                            'ignore')
            gamesInfoVisId = unicodedata.normalize('NFKD', gamesInfoVis['id'][0][0][0]).encode('ascii', 'ignore')

            try:
                team_name_home_round_dict = Team_name_dict.get(gamesInfoHomeName)
                round_num = len(team_name_home_round_dict.keys())
                team_name_home_round_dict.update({round_num: gamesInfoId})
                Team_name_dict.update({gamesInfoHomeName: team_name_home_round_dict})
            except:
                team_name_home_round_dict = {1: gamesInfoId}
                team_name_home_round_dict.update({'Id': gamesInfoHomeId})
                Team_name_dict.update({gamesInfoHomeName: team_name_home_round_dict})

            try:
                team_name_vis_round_dict = Team_name_dict.get(gamesInfoVisName)
                round_num = len(team_name_vis_round_dict.keys())
                team_name_vis_round_dict.update({round_num: gamesInfoId})
                Team_name_dict.update({gamesInfoVisName: team_name_vis_round_dict})
            except:
                team_name_vis_round_dict = {1: gamesInfoId}
                team_name_vis_round_dict.update({'Id': gamesInfoVisId})
                Team_name_dict.update({gamesInfoVisName: team_name_vis_round_dict})

        # for key in Team_name_dict.keys():
        #     print (key, Team_name_dict.get(key))
        return Team_name_dict

    def read_gameId_directory(self):
        gameId_directory_dir = "../player_statistic/gameId directory.csv"
        gameId_directory_list = []
        with open(gameId_directory_dir) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                gameId = row.get("gameId")
                directory = row.get("directory")
                gameId_directory_list.append([gameId, directory])
        return gameId_directory_list

    def find_game_dir(self, target_gameId):
        gameId_directory_list = self.read_gameId_directory()
        for gameId_directory in gameId_directory_list:
            if target_gameId == gameId_directory[0]:
                # print gameId_directory[1].split(".")[0]
                return gameId_directory[1].split(".")[0]

        raise ValueError("can't find target_gameId")

    def compute_correlated_coefficient(self, csv_read_dict_list):
        coe_target_dict = {'value': [], 'value all': [], 'G': [], 'A': [], 'P': []}

        coe_save_value_list = []

        for csv_row_read_dict_index in range(0, len(csv_read_dict_list)):
            csv_row_read_dict = csv_read_dict_list[csv_row_read_dict_index]
            for key in coe_target_dict.keys():
                value = float(csv_row_read_dict.get(key))
                value_new = (coe_target_dict.get(key))
                value_new.append(value)
                coe_target_dict.update({key: value_new})

        coe_save_value_list.append(pearsonr(coe_target_dict.get('value'),
                                            coe_target_dict.get('G'))[0])
        coe_save_value_list.append(pearsonr(coe_target_dict.get('value'),
                                            coe_target_dict.get('A'))[0])
        coe_save_value_list.append(pearsonr(coe_target_dict.get('value'),
                                            coe_target_dict.get('P'))[0])
        coe_save_value_list.append(pearsonr(coe_target_dict.get('value'),
                                            coe_target_dict.get('value all'))[0])

        return coe_save_value_list

    def draw_round_by_round_coe(self, coefficient_record):
        goal_coe = coefficient_record[:, 0]
        assist_coe = coefficient_record[:, 1]
        point_coe = coefficient_record[:, 2]
        value_coe = coefficient_record[:, 3]
        rounds = range(1, self.ROUND_NUMBER + 1)
        plt.plot(rounds, goal_coe, label=r'$\rho(GIM^{round}, Goals)$')
        plt.plot(rounds, assist_coe, label=r'$\rho(GIM^{round}, Assists)$')
        plt.plot(rounds, point_coe, label=r'$\rho(GIM^{round}, Points)$')
        plt.plot(rounds, value_coe, label=r'$\rho(GIM^{round}, GMI^{season})$')
        plt.legend(loc='lower right')
        plt.title("Round by Round Correlation in 2015-2016 NHL season", fontsize=14)
        plt.xlabel("Round", fontsize=14)
        plt.ylabel("Correlation", fontsize=14)
        # plt.show()
        plt.savefig("../figures/GMI_round_by_round_coe.png")

    def write_round_correlation(self, coefficient_record,
                                csv_name):
        csv_name = './game_impact_metric_sta/GIM_round_correlation-MTL{0}.csv'.format(self.MAX_TRACE_LENGTH)
        standard_statistic_fields = ['assistant', 'goal', 'point', 'GIM']
        with open(csv_name, 'wb') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(standard_statistic_fields)
            for line in coefficient_record:
                wr.writerow([line[1], line[0], line[2], line[3]])


if __name__ == '__main__':
    GIM = GoalImpactMetric()
    player_Info = GIM.read_players_info()
    playerId_skateInfo_dict = GIM.combine_playerId_to_skate_info(player_Info)

    Team_name_dict = GIM.team_match_statistic()

    if GIM.IS_DIFFERENCE:
        GIM.aggregate_diff_values()
    else:
        GIM.aggregate_values()

    coefficient_record = []

    for round_num in range(1, GIM.ROUND_NUMBER + 1):

        game_compute_dict_list = []

        for key in Team_name_dict.keys():
            team_sta = Team_name_dict.get(key)
            teamId = team_sta.get("Id")
            gameId = team_sta.get(round_num)
            game_compute_dict_list.append({"gameId": gameId, "team id": teamId})

        for game_compute_dict in game_compute_dict_list:
            gameId_target = game_compute_dict.get("gameId")
            teamId_target = game_compute_dict.get("team id")
            game_target_dir = GIM.find_game_dir(gameId_target)

            if GIM.IS_DIFFERENCE:
                GIM.aggregate_match_diff_values(game_target_dir, teamId_target)
            else:
                GIM.aggregate_match_values(game_target_dir, teamId_target)
            if GIM.IS_POSSIBILITY:
                possi_write = "_possibility"
            else:
                possi_write = ""
            if GIM.IS_DIFFERENCE:
                diff_write = GIM.DIFFERENCE_TYPE
            else:
                diff_write = ""
                # break
        player_value_dict_list = GIM.transfer_save_format_by_match(playerId_skateInfo_dict)
        coefficient = GIM.compute_correlated_coefficient(player_value_dict_list)
        coefficient_record.append(np.asarray(coefficient))
    GIM.draw_round_by_round_coe(np.asarray(coefficient_record))
    GIM.write_round_correlation(coefficient_record=coefficient_record)

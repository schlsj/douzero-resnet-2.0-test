import builtins
import multiprocessing as mp
import os.path
import pickle
import copy
from game_eval import GameEnv


EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {3: '3', 4: '4', '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

output_to_file = False
output_list = []


def print(*args, **kwargs):
    builtins.print(*args, **kwargs)
    if output_to_file:
        end = "\n"
        if kwargs.get("end") is not None:
            end = kwargs.get("end")
        output_list.append(" ".join(args) + end)


def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ['landlord', 'landlord_down', 'landlord_up']:
        if card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        elif card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        else:
            from .deep_agent import DeepAgent
            if not isinstance(card_play_model_path_dict[position], list):
                players[position] = DeepAgent(
                    position, card_play_model_path_dict[position])
            else:
                paths = card_play_model_path_dict[position]
                players[position] = {
                    "landlord": DeepAgent("landlord", paths[0]),
                    "landlord_down": DeepAgent("landlord_down", paths[1]),
                    "landlord_up": DeepAgent("landlord_up", paths[2]),
                }
    return players


def print_card(cards, end="\n"):
    print("".join(EnvCard2RealCard[card] for card in cards), end=end)


def get_modelname_by_path(model_path):
    sep = "/"
    if "|" in model_path:
        model_path = model_path.split("|")
    if isinstance(model_path, list):
        mlist = []
        for path in model_path:
            if "\\" in path:
                sep = "\\"
            mlist.append(path.split(sep)[-1].split(".")[0])
        return "|".join(mlist)
    else:
        if "\\" in model_path:
            sep = "\\"
        return model_path.split(sep)[-1].split(".")[0]


def mp_simulate(card_play_data_list, card_play_model_path_dict, q):
    for k in card_play_model_path_dict:
        if "|" in card_play_model_path_dict[k]:
            card_play_model_path_dict[k] = card_play_model_path_dict[k].split(
                "|")
    players = load_card_play_models(card_play_model_path_dict)

    Env = GameEnv(players)
    enable_output = False

    if enable_output:
        print("对局模型信息：")
        for position in ['landlord', 'landlord_down', 'landlord_up']:
            print("{}：{}".format(position, get_modelname_by_path(
                card_play_model_path_dict[position])))

    def start_game(env, idx, card_play_data):
        _card_play_data = copy.deepcopy(card_play_data)

        # No bid phase: directly set up card play state
        env.three_landlord_cards = _card_play_data['three_landlord_cards']

        # Assign hands: first->landlord, second->landlord_down, third->landlord_up
        landlord_cards = sorted(
            _card_play_data['first'] + _card_play_data['three_landlord_cards'])
        landlord_down_cards = sorted(_card_play_data['second'])
        landlord_up_cards = sorted(_card_play_data['third'])

        env.info_sets['landlord'].player_hand_cards = landlord_cards
        env.info_sets['landlord_down'].player_hand_cards = landlord_down_cards
        env.info_sets['landlord_up'].player_hand_cards = landlord_up_cards

        # Set game state to skip bid
        env.bid_over = True
        env.draw = False
        env.bid_count = 1
        env.bid_info = [1, 0, 0]
        for pos in ['landlord', 'landlord_down', 'landlord_up']:
            env.info_sets[pos].bid_over = True
            env.info_sets[pos].bid_count = 1

        # Initialize acting player and infoset
        env.get_acting_player_position()
        env.game_infoset = env.get_infoset()

        if enable_output:
            print_card(env.info_sets["landlord"].player_hand_cards)
            print_card(env.info_sets["landlord_down"].player_hand_cards)
            print_card(env.info_sets["landlord_up"].player_hand_cards)

        step_index = 0
        while not env.game_over:
            step_index += 1
            action, action_list = env.step()
            if enable_output:
                if action:
                    print("".join(EnvCard2RealCard[card]
                          for card in action), end=" ")
                else:
                    print("Pass", end=" ")
                if step_index % 3 == 0:
                    print()

        env.reset()

    for idx, card_play_data in enumerate(card_play_data_list):
        if enable_output:
            print("\n----- 第%d局 -----" % (idx + 1))
        start_game(Env, idx, card_play_data)

    q.put((Env.num_wins['landlord'],
           Env.num_wins['farmer'],
           Env.num_scores['landlord'],
           Env.num_scores['farmer'],
           ))


def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker


def evaluate(landlord, landlord_down, landlord_up, eval_data, num_workers):

    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_down': landlord_down,
        'landlord_up': landlord_up,
    }
    model_name_dict = {
        'landlord': get_modelname_by_path(landlord),
        'landlord_down': get_modelname_by_path(landlord_down),
        'landlord_up': get_modelname_by_path(landlord_up),
    }

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_paly_data in card_play_data_list_each_worker:
        p = ctx.Process(
            target=mp_simulate,
            args=(card_paly_data, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]

    num_total_wins = num_landlord_wins + num_farmer_wins
    print("\n对局模型信息：")
    for position in ['landlord', 'landlord_down', 'landlord_up']:
        print("{}：{}".format(position, model_name_dict[position]))
    print('WP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_wins /
          num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_scores /
          num_total_wins, num_farmer_scores / num_total_wins))

    if output_to_file:
        eval_name = model_name_dict['landlord'] + '__' + \
            model_name_dict['landlord_down'] + \
            '__' + model_name_dict['landlord_up']
        if not os.path.exists("./eval_results"):
            os.mkdir("./eval_results")
        with open("./eval_results/" + eval_name + ".txt", 'a') as f:
            f.write("".join(output_list))

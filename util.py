from keras.layers import Input, Dense, Concatenate, Flatten
from keras import Sequential
from keras.models import Model
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from keras.optimizers.legacy import Adam
import pk_calc

SIZE_FIELD = pk_calc.SIZE_FIELD
SIZE_BODY_PKMN = pk_calc.SIZE_BODY_PKMN 
SIZE_MOVE = pk_calc.SIZE_MOVE
SIZE_PKMN = SIZE_BODY_PKMN + SIZE_MOVE*4


def create_model(n_action) -> Model:
    input_l = Input(shape= (1777,))
    #input_field = input_l[:, :SIZE_FIELD]
    #
    #offset_status_pkmn = SIZE_FIELD + SIZE_BODY_PKMN
    #offset_pkmn = SIZE_FIELD + SIZE_PKMN
    #input_status_pkmn = input_l[:, SIZE_FIELD:offset_status_pkmn] #70
    #input_moves_pkmn = input_l[:, offset_status_pkmn: offset_pkmn] #228
    #
    #offset_status_opp_pkmn = offset_pkmn + SIZE_BODY_PKMN
    #offset_opp_pkmn = offset_pkmn + SIZE_PKMN
    #input_status_opp_pkmn = input_l[:, offset_pkmn: offset_status_opp_pkmn]
    #input_moves_opp_pkmn = input_l[:, offset_status_opp_pkmn: offset_opp_pkmn]
    #
    #offset_team = offset_opp_pkmn + SIZE_PKMN *5
    #input_team = input_l[:, offset_opp_pkmn:offset_team]
    #
    #offset_opp_team = offset_team + SIZE_PKMN *5
    #input_opp_team = input_l[:, offset_team: offset_opp_team]
    #
    #fainted_team = Flatten()(input_l[:, -2])
    #fainted_opp_team = Flatten()(input_l[:, -1])
    #
    #
    #l_field_m_pkmn = Dense(8, activation="relu")(Concatenate()([input_field, input_moves_pkmn]))
    #l_field_m_opp_pkmn = Dense(8, activation="relu")(Concatenate()([input_field, input_moves_opp_pkmn]))
    #
    #l_pkmn = Dense(32, activation="relu")(Concatenate()([input_status_pkmn, input_moves_pkmn]))
    #l_opp_pkmn = Dense(32, activation="relu")(Concatenate()([input_status_opp_pkmn, input_moves_opp_pkmn]))
    #
    #l_m_opp_pkmn = Dense(32, activation="relu")(Concatenate()([input_status_opp_pkmn, input_moves_pkmn]))
    #l_opp_m_pkmn = Dense(32, activation="relu")(Concatenate()([input_status_pkmn, input_moves_opp_pkmn]))

    #hidden_layer_1 = Dense(256, activation="relu")(Concatenate()([
    #    l_field_m_pkmn,
    #    l_field_m_opp_pkmn,
    #    l_pkmn,
    #    l_opp_pkmn,
    #    l_m_opp_pkmn,
    #    l_opp_m_pkmn,
    #    input_team,
    #    input_opp_team,
    #    fainted_team,
    #    fainted_opp_team
    #]))
    #output_layer = Dense(n_action, activation="softmax")(hidden_layer_1)
    #model = Model(inputs=input_l, outputs=output_layer)  
    
    model = Sequential([input_l, 
                        Dense(256, activation="relu"), 
                        Dense(256, activation="relu"),
                        Dense(n_action, activation="softmax")])  
  
    return model

def create_agent(model, n_action, l_policy = 10000, random_step = 1000):
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=l_policy,
    )

    dqn_agent = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=random_step,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn_agent.compile(Adam(learning_rate=0.0001), metrics=["mae"])

    return dqn_agent
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.optimizers.legacy import Adam

SIZE_FIELD = 12
SIZE_STATUS_PKMN = 23
SIZE_MOVE = 25
SIZE_PKMN = 23 + 25*4
def create_model(n_action) -> Model:
    input_l = Input(shape= (1490,))
    input_field = input_l[:, :SIZE_FIELD]

    offset_status_pkmn = SIZE_FIELD + SIZE_STATUS_PKMN
    offset_pkmn = SIZE_FIELD + SIZE_PKMN
    input_status_pkmn = input_l[:, SIZE_FIELD:offset_status_pkmn]
    input_moves_pkmn = input_l[:, offset_status_pkmn: offset_pkmn]
    
    offset_status_opp_pkmn = offset_pkmn + SIZE_STATUS_PKMN
    offset_opp_pkmn = offset_pkmn + SIZE_PKMN
    input_status_opp_pkmn = input_l[:, offset_pkmn: offset_status_opp_pkmn]
    input_moves_opp_pkmn = input_l[:, offset_status_opp_pkmn: offset_opp_pkmn]

    offset_team = offset_opp_pkmn + SIZE_PKMN *5
    input_team = input_l[:, offset_opp_pkmn:offset_team]

    offset_opp_team = offset_team + SIZE_PKMN *5
    input_opp_team = input_l[:, offset_team: offset_opp_team]

    fainted_team = Flatten()(input_l[:, -2])
    fainted_opp_team = Flatten()(input_l[:, -1])

    
    l_field_m_pkmn = Dense(8, activation="relu")(Concatenate()([input_field, input_moves_pkmn]))
    l_field_m_opp_pkmn = Dense(8, activation="relu")(Concatenate()([input_field, input_moves_opp_pkmn]))

    l_pkmn = Dense(32, activation="relu")(Concatenate()([input_status_pkmn, input_moves_pkmn]))
    l_opp_pkmn = Dense(32, activation="relu")(Concatenate()([input_status_opp_pkmn, input_moves_opp_pkmn]))

    l_m_opp_pkmn = Dense(32, activation="relu")(Concatenate()([input_status_opp_pkmn, input_moves_pkmn]))
    l_opp_m_pkmn = Dense(32, activation="relu")(Concatenate()([input_status_pkmn, input_moves_opp_pkmn]))

    hidden_layer_1 = Dense(256, activation="relu")(Concatenate()([
        l_field_m_pkmn,
        l_field_m_opp_pkmn,
        l_pkmn,
        l_opp_pkmn,
        l_m_opp_pkmn,
        l_opp_m_pkmn,
        input_team,
        input_opp_team,
        fainted_team,
        fainted_opp_team
    ]))
    output_layer = Dense(n_action, activation="softmax")(hidden_layer_1)
    model = Model(inputs=input_l, outputs=output_layer)    
    return model

def create_agent(model, n_action):
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn_agent = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=5,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn_agent.compile(Adam(learning_rate=0.0001), metrics=["mae"])

    return dqn_agent
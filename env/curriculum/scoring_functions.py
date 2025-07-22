def ball_touched(env, episode, prev_episode_chunks):
    ball_touches = [car.ball_touches > 0 for car in env.state.cars.values()]
    return float(any(ball_touches))


def goal_scored(env, episode, prev_episode_chunks):
    return float(env.state.goal_scored)

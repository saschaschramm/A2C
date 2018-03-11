from utilities import init_env

def main():
    map_name = "4x4"
    env = init_env(map_name)

    observation = env.reset()
    print("observation1", observation)

    observation = env.step(2)
    print("observation2", observation)


if __name__ == '__main__':
    main()
import argparse

def create_parser_main_pattern():
    parent_parser = argparse.ArgumentParser()
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--save_dir', type=str, help='save_dir', default='../Simu')
    main_parser.add_argument('--name', type=str, help='Name', default='local/test')
    main_parser.add_argument('--P', type=int, help='Number of neurons', default=10)
    main_parser.add_argument('--neuronClass', type=str, help='neuronClass', default='MSN_IAF_EXP')
    main_parser.add_argument('--Apostpre', type=float, help='Apostpre', default=1.)
    main_parser.add_argument('--Aprepost', type=float, help='Aprepost', default=-1.)
    main_parser.add_argument('--homeostasy', type=float, help='homeostasy', default=0.9)
    main_parser.add_argument('--epsilon', type=float, help='epsilon', default=0.02)
    main_parser.add_argument('--noise_stim', type=float, help='noise_stim', default=0.)
    main_parser.add_argument('--noise_input', type=float, help='noise_input', default=0.)
    main_parser.add_argument('--noise_pattern', type=float, help='noise_input', default=0.)
    main_parser.add_argument('--stop_learning', type=str, help='stop_learning', default='None')
    main_parser.add_argument('--num_success_params', type=int, help='num_success_params', default=0)
    main_parser.add_argument('--dt', type=float, help='dt', default=0.2)
    main_parser.add_argument('--num_training', type=int, help='num_training', default=50)
    main_parser.add_argument('--stim_duration', type=float, help='stim_duration', default=50.)
    main_parser.add_argument('--stim_offset', type=float, help='stim_offset', default=20.)
    main_parser.add_argument('--save', help='save', action='store_true', default=False)
    main_parser.add_argument('--plot', help='plot', action='store_true', default=False)
    main_parser.add_argument('--random_seed', type=int, help='random_seed', default=None)

    network_subparsers = parent_parser.add_subparsers(title='network', dest='network')

    single_parser = network_subparsers.add_parser('single', help='single', add_help=False, parents=[main_parser])

    dual_parser_args = argparse.ArgumentParser(add_help=False)
    dual_parser_args.add_argument('--J_matrix', type=int, help='J_matrix', default=0)
    dual_parser_args.add_argument('--J_value', type=str, help='J_value', default='0.4')
    dual_parser_args.add_argument('--J_reward', type=str, help='J_reward', default='differential')
    dual_parser = network_subparsers.add_parser('dual', help='dual', add_help=False,
                                                parents=[main_parser, dual_parser_args])

    for parser, parser_args in [(single_parser, None), (dual_parser, dual_parser_args)]:
        pattern_subparser = parser.add_subparsers(title='pattern', dest='pattern')
        list_parent_parsers = [main_parser]
        if parser_args is not None:
            list_parent_parsers.append(parser_args)
        list_pattern_parser = pattern_subparser.add_parser('list_pattern', help='list_pattern', add_help=False,
                                                           parents=list_parent_parsers)
        list_pattern_parser.add_argument('--stim_by_pattern', type=int, help='stim_by_pattern', default=3)
        list_pattern_parser.add_argument('--repartition', type=str, help='repartition', default='uniform_stim')
        list_pattern_parser.add_argument('--p_reward', type=float, help='p_reward', default=0.5)
        list_pattern_parser.add_argument('--stim_delay', type=float, help='stim_delay', default=1.)
        list_pattern_parser.add_argument('--num_simu', type=int, help='num_simu', default=10)
        jitter_parser = pattern_subparser.add_parser('jitter', help='jitter', add_help=False,
                                                     parents=list_parent_parsers)
        jitter_parser.add_argument('--stim_by_pattern', type=int, help='stim_by_pattern', default=3)
        jitter_parser.add_argument('--repartition', type=str, help='repartition', default='uniform_stim')
        jitter_parser.add_argument('--p_reward', type=float, help='p_reward', default=0.5)
        jitter_parser.add_argument('--stim_delay', type=float, help='stim_delay', default=1.)
        jitter_parser.add_argument('--num_simu', type=int, help='num_simu', default=10)
        succession_parser = pattern_subparser.add_parser('succession', help='succession', add_help=False,
                                                         parents=list_parent_parsers)
        succession_parser.add_argument('--stim_delay', type=float, help='stim_delay', default=1.)
        poisson_pattern_parser = pattern_subparser.add_parser('poisson', help='poisson', add_help=False,
                                                              parents=list_parent_parsers)
        poisson_pattern_parser.add_argument('--p_reward', type=float, help='p_reward', default=0.5)
        poisson_pattern_parser.add_argument('--num_simu', type=int, help='num_simu', default=100)
        poisson_pattern_parser.add_argument('--duration_poisson', type=float, help='duration_poisson', default=1.)
        poisson_pattern_parser.add_argument('--noise_poisson', type=float, help='noise_poisson', default=1.)
        example_pattern_parser = pattern_subparser.add_parser('example', help='example', add_help=False,
                                                              parents=list_parent_parsers)
        example_pattern_parser.add_argument('--no_reward', type=int, help='no_reward', default=0)
        example_pattern_parser.add_argument('--num_simu', type=int, help='num_simu', default=10)
        example_pattern_parser.add_argument('--pattern_example', type=str, help='pattern_example', default='A')
        example_pattern_parser.add_argument('--start_weight', type=str, help='start_weight', default='low')

    return parent_parser

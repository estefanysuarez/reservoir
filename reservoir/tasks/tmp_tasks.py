
def run_pattrn_recog(target, reservoir_states, time_lens, normalize=False, **kwargs):

    if normalize: reservoir_states = (reservoir_states - reservoir_states.mean(axis=1)[:,np.newaxis])

    sections = [np.sum(time_lens[:idx]) for idx in range(1, len(time_lens))]
    X = np.split(reservoir_states.squeeze(), sections, axis=0)
    Y = np.split(target, sections, axis=0)

    train_frac = 0.9
    n_samples = len(X)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))

    x_train = np.vstack(X[:n_train_samples])
    y_train = np.vstack(Y[:n_train_samples])

    x_test = np.vstack(X[n_train_samples:])
    y_test = Y[n_train_samples:]

    ridge_multi_regr = MultiOutputRegressor(Ridge(fit_intercept=True, normalize=False, solver='auto', alpha=0.0))
    Y_pred = np.split(ridge_multi_regr.fit(x_train, y_train).predict(x_test), np.array(sections[n_train_samples:]), axis=0)

    #lr_multi_regr = MultiOutputRegressor(LinearRegression(fit_intercept=True, normalize=False, copy_X = True))
    #Y_pred = np.split(lr_multi_regr.fit(x_train, y_train).predict(x_test), np.array(pattern_lens[train_samples:-1]), axis=0)

    Y_test_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in y_test])
    print(Y_test_mean)

    Y_pred_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in Y_pred])
    print(Y_pred_mean)

    with np.errstate(divide='ignore', invalid='ignore'):
        cm = metrics.confusion_matrix(Y_test_mean, Y_pred_mean)
        cm_norm = np.diagonal(cm)/np.sum(cm, axis=1)
        perf = len(np.where(cm_norm > 0.9)[0])

        # print('\n---------' + str(perf))

    return perf


def run_nonlin_cap(X, Y, OMEGA=None, normalize=False, ax=None, **kwargs):
    """
    In this task, the linear readout is required to replay a noninear version of
    the input sequence s.
    """
    # OMEGA: level of nonlinearity required by the task
    if OMEGA is None: OMEGA = get_default_task_params('nonlin_cap')

    # get train and test sets
    x_train, x_test = X
    y_train, y_test = Y

    # define transient
    transient = 0 #number of initial time points to discard

    if (x_train.squeeze().ndim == 2) and (x_test.ndim == 2):
        x_train = x_train.squeeze()[transient:,:]
        x_test  = x_test.squeeze()[transient:,:]

    else:
        x_train = x_train.squeeze()[transient:, np.newaxis]
        x_test  = x_test.squeeze()[transient:, np.newaxis]

    y_train = y_train.squeeze()[transient:]
    y_test  = y_test.squeeze()[transient:]

    if normalize:
        x_train = (x_train - x_train.mean(axis=1)[:,np.newaxis]).squeeze()
        x_test  = (x_test - x_test.mean(axis=1)[:,np.newaxis]).squeeze()

    res = []
    for omega in OMEGA:

       model = LinearRegression(fit_intercept=False, normalize=False).fit(x_train[1:], np.sin(omega*y_train[:-1]))
       y_pred =  model.predict(x_test[1:])

       with np.errstate(divide='ignore', invalid='ignore'):
           perf = np.abs(np.corrcoef(np.sin(omega*y_test[:-1]), y_pred)[0][1])
           # print(perf)

           # if perf > 0.85:
           #     print('\n----------------------')
           #     print(' Omega = ' + str(np.log10(omega)))
           #     print(' perf  =   ' + str(perf))
           #
           #     plt.scatter(np.sin(omega*y_test[:-1]), y_pred)
           #     plt.show()
           #     plt.close()

       # save results
       res.append(perf)

    return np.array(res), np.log10(OMEGA)


def run_fcn_app(X, Y, TAU=None, OMEGA=None, normalize=False, ax=None, **kwargs):

    # define nonlinearity and memory capcity degree of the fcn_approx
    if TAU is None:   TAU   = get_default_task_params('fcn_app')[0]
    if OMEGA is None: OMEGA = get_default_task_params('fcn_app')[1]

    param_space = dict(tau = TAU, omega = OMEGA)
    param_grid = list(ParameterGrid(param_space))

    # get train and test sets
    x_train, x_test = X
    y_train, y_test = Y

    # define transient
    transient = 0 #number of initial time points to discard

    if (x_train.squeeze().ndim == 2) and (x_test.ndim == 2):
        x_train = x_train.squeeze()[transient:,:]
        x_test  = x_test.squeeze()[transient:,:]

    else:
        x_train = x_train.squeeze()[transient:, np.newaxis]
        x_test  = x_test.squeeze()[transient:, np.newaxis]

    y_train = y_train.squeeze()[transient:]
    y_test  = y_test.squeeze()[transient:]

    if normalize:
        x_train = (x_train - x_train.mean(axis=1)[:,np.newaxis]).squeeze()
        x_test  = (x_test - x_test.mean(axis=1)[:,np.newaxis]).squeeze()

    res  = np.zeros((len(param_space['tau']), len(param_space['omega'])))
    for params_set in param_grid:

        # level of nonlinearity and temporal memory capacity required
        # by the fcn_approx task
        omega = params_set['omega']
        tau = params_set['tau']

        model = LinearRegression(fit_intercept=False, normalize=False).fit(x_train[tau:], np.sin(omega*y_train[:-tau]))
        y_pred =  model.predict(x_test[tau:])

        with np.errstate(divide='ignore', invalid='ignore'):
            perf = np.abs(np.corrcoef(np.sin(omega*y_test[:-tau]), y_pred)[0][1])
            # print(perf)

            # if perf > 0.85:
            #     print('\n----------------------')
            #     print(' Tau  = ' + str(tau))
            #     print(' Omega = ' + str(np.log10(omega)))
            #     print(' perf   =   ' + str(perf))
            #
            #     plt.scatter(np.sin(omega*y_test[:-tau]), y_pred)
            #     plt.show()
            #     plt.close()


        # print out results
        coord_omega = np.where(param_space['omega'] == omega)[0][0]
        coord_tau   = np.where(param_space['tau'] == tau)[0][0]
        res[coord_tau, coord_omega] = perf

    return res, (param_space['tau'], np.log10(param_space['omega']))



#%% --------------------------------------------------------------------------------------------------------------------
# TASKS INPUTS/OUTPUTS
# ----------------------------------------------------------------------------------------------------------------------
# def writeDict(dict, filename, sep):
#     with open(filename, "a") as f:
#         for k, v in dict.iteritems():
#             f.write(k + sep + str(v) + '\n')
#     f.close()
#
#
# def readDict(filename, sep):
#
#     def is_number(s):
#         try:
#             float(s)
#             return True
#         except ValueError:
#             pass
#
#         try:
#             import unicodedata
#             unicodedata.numeric(s)
#             return True
#         except (TypeError, ValueError):
#             pass
#
#         return False
#
#     with open(filename, "r") as f:
#         dict = {}
#
#         for line in f:
#             k, v = line.split(sep)
#
#             print(is_number(v))
#
#             if is_number(v): dict[k] = np.float(v)
#
#             else: dict[k] = v
#
#     f.close()
#     return(dict)

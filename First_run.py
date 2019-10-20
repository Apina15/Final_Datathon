import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
df = pd.read_csv('No_na.csv')



def results_disp(name_, scores, position_num, test_results):
    print(name_ + ': %0.3f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    test_results.insert(position_num, name_, [scores.mean()], True)
    test_results.insert(position_num, name_ + 'std', [scores.std() * 2], True)

X = df.drop(columns=['target'])
y = df['target'].values


def classifier_run_ranking(cv_num, ranking_bool, algo):
    test_results = pd.DataFrame()  # This is a blank data frame that will keep track of the results.
    # The following gets the name of the classifier.
    algo_name = algo.__class__.__name__

    # This list tells the function which scoring results to get.
    scoring_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scores = cross_validate(algo, X, y, cv=cv_num, scoring=scoring_list, return_estimator=True)

    acc_score = scores['test_accuracy']
    prec_score = scores['test_precision']
    f1_score = scores['test_f1']
    recall_score = scores['test_recall']
    roc_score = scores['test_roc_auc']

    # The following displays the average of the scores and standard deviation as well as saving it on the data frame.
    results_disp('Accuracy', acc_score, 0, test_results)
    results_disp('Precision', prec_score, 2, test_results)
    results_disp('f1', f1_score, 4, test_results)
    results_disp('Recall', recall_score, 6, test_results)
    results_disp('ROC', roc_score, 8, test_results)

    test_results.to_csv("Test_results_ " + algo_name + '.csv', index=False)
    del test_results

    if ranking_bool:

        for idx, estimator in enumerate(scores['estimator']):
            print("Features sorted by their score for estimator {}:".format(idx))
            feature_importance = pd.DataFrame(estimator.feature_importances_, index=list(X.columns),
                                              columns=['importance']).sort_values('importance', ascending=False)
            if idx == 0:
                total_feat_imp = pd.DataFrame(estimator.feature_importances_, index=list(X.columns),
                                              columns=['importance'])
            else:
                total_feat_imp += pd.DataFrame(estimator.feature_importances_, index=list(X.columns),
                                               columns=['importance'])
            print(feature_importance)

        total_feat_imp = total_feat_imp.sort_values('importance', ascending=False) / cv_num
        print('This is the average of the feature importance\n')
        print(total_feat_imp)
        total_feat_imp.to_csv('Feature_ranking_' + algo_name + '.csv')
        del total_feat_imp




classifier_run_ranking(10, True,  RandomForestClassifier(random_state=0, n_estimators=100))
classifier_run_ranking(10, True,  AdaBoostClassifier())
classifier_run_ranking(10, True,  DecisionTreeClassifier(max_depth=3))
classifier_run_ranking(10, True,  ExtraTreesClassifier(n_estimators=100, random_state=0))
classifier_run_ranking(10, True,  GradientBoostingClassifier())
classifier_run_ranking(10, False,  BaggingClassifier())

using Test
import CategoricalArrays
import CategoricalArrays.categorical
using MLJBase
using StableRNGs
using Random
using Tables
import MLJTestInterface
using StatisticalMeasures

# load code to be tested:
import DecisionTree
using MLJDecisionTreeInterface

Random.seed!(1234)

@testset "generic interface tests" begin
    @testset "regressors" begin
        failures, summary = MLJTestInterface.test(
            [DecisionTreeRegressor, RandomForestRegressor],
            MLJTestInterface.make_regression()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false, # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "classifiers" begin
        for data in [
            MLJTestInterface.make_binary(),
            MLJTestInterface.make_multiclass(),
        ]
            failures, summary = MLJTestInterface.test(
                [DecisionTreeClassifier, RandomForestClassifier, AdaBoostStumpClassifier],
                data...;
                mod=@__MODULE__,
                verbosity=0, # bump to debug
                throw=false, # set to true to debug
            )
            @test isempty(failures)
        end
    end
end

stable_rng() = StableRNGs.StableRNG(123)

# get some test data:
Xraw, yraw = @load_iris
X = Tables.matrix(Xraw);
y = int(yraw);
_classes = MLJDecisionTreeInterface.classes(yraw)
features = MLJDecisionTreeInterface._columnnames(Xraw)

baretree = DecisionTreeClassifier(rng=stable_rng())

baretree.max_depth = 1
fitresult, cache, report = MLJBase.fit(baretree, 2, X, y, features, _classes);
baretree.max_depth = -1 # no max depth
fitresult, cache, report =
    MLJBase.fit(baretree, 1, X, y, features, _classes);

# in this case decision tree is a perfect predictor:
yhat = MLJBase.predict_mode(baretree, fitresult, X);
@test int(yhat) == y

# but pruning upsets this:
baretree.post_prune = true
baretree.merge_purity_threshold=0.1
fitresult, cache, report =
    MLJBase.fit(baretree, 2, X, y, features, _classes)
yhat = MLJBase.predict_mode(baretree, fitresult, X);
@test int(yhat) != y
yhat = MLJBase.predict(baretree, fitresult, X);

# check preservation of levels:
yyhat = predict_mode(baretree, fitresult, X[1:3, :])
@test MLJBase.classes(yyhat[1]) == MLJBase.classes(yraw)

# check report and fitresult fields:
@test Set([:classes_seen, :print_tree, :features]) == Set(keys(report))
@test Set(report.classes_seen) == Set(levels(yraw))
@test report.print_tree(2) === nothing # :-(
@test report.features == [:sepal_length, :sepal_width, :petal_length, :petal_width]

fp = fitted_params(baretree, fitresult)
@test Set([:tree, :encoding, :features, :raw_tree]) == Set(keys(fp))
@test fp.features == report.features
enc = fp.encoding
@test Set(values(enc)) == Set(["virginica", "setosa", "versicolor"])
@test enc[y[end]] == "virginica"

using Random: seed!
seed!(0)

n,m = 10^3, 5;
X = rand(stable_rng(), n,m);
features = [:x1, :x2, :x3, :x4, :x5]
weights = rand(stable_rng(), -1:1,m);
y = X * weights;

R1Tree = DecisionTreeRegressor(
    min_samples_leaf=5,
    merge_purity_threshold=0.1,
    rng=stable_rng(),
)
R2Tree = DecisionTreeRegressor(min_samples_split=5, rng=stable_rng())
model1, = MLJBase.fit(R1Tree,1, X, y, features)

vals1 = MLJBase.predict(R1Tree,model1,X)
R1Tree.post_prune = true
model1_prune, = MLJBase.fit(R1Tree,1, X, y, features)
vals1_prune = MLJBase.predict(R1Tree,model1_prune,X)
@test vals1 !=vals1_prune

@test DecisionTree.R2(y, vals1) > 0.8

model2, = MLJBase.fit(R2Tree, 1, X, y, features)
vals2 = MLJBase.predict(R2Tree, model2, X)
@test DecisionTree.R2(y, vals2) > 0.8


## TEST ON ORDINAL FEATURES OTHER THAN CONTINUOUS

N = 20
Xraw = (
    x1=rand(stable_rng(),N),
    x2=categorical(rand(stable_rng(), "abc", N), ordered=true),
    x3=collect(1:N),
);
y1 = Xraw.x2;
y2 = float.(Xraw.x3);

rgs = DecisionTreeRegressor(rng=stable_rng())
X, y, features = MLJBase.reformat(rgs, Xraw, y2)

fitresult, _, _ = MLJBase.fit(rgs, 1, X, y, features)
@test rms(predict(rgs, fitresult, X), y) < 1.5

clf = DecisionTreeClassifier()
X, y, features, _classes = MLJBase.reformat(clf, Xraw, y1)

fitresult, _, _ = MLJBase.fit(clf, 1, X, y, features, _classes)
@test sum(predict(clf, fitresult, X) .== y1) == 0 # perfect prediction


# ENSEMBLES AND INTEGRATION WITH MLJBASE MACHINES

rfc = RandomForestClassifier(rng=stable_rng())
abs = AdaBoostStumpClassifier(rng=stable_rng())

X, y = MLJBase.make_blobs(100, 3; rng=stable_rng())

m = machine(rfc, X, y)
fit!(m)
@test accuracy(predict_mode(m, X), y) > 0.95

m = machine(abs, X, y)
fit!(m)
@test accuracy(predict_mode(m, X), y) > 0.95

X, y = MLJBase.make_regression(rng=stable_rng())
rfr = RandomForestRegressor(rng=stable_rng())
m = machine(rfr, X, y)
fit!(m)
@test rms(predict(m, X), y) < 0.4

N = 10
function reproducibility(model, X, y, loss)
    if !(model isa AdaBoostStumpClassifier)
        model.n_subfeatures = 1
    end
    mach = machine(model, X, y)
    train, test = partition(eachindex(y), 0.7)
    errs = map(1:N) do i
        model.rng = stable_rng()
        fit!(mach, rows=train, force=true, verbosity=0)
        yhat = predict(mach, rows=test)
        loss(yhat, y[test]) |> mean
    end
    return length(unique(errs)) == 1
end

@testset "reporoducibility" begin
    X, y = make_blobs(rng=stable_rng());
    loss = BrierLoss()
    for model in [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostStumpClassifier(),
    ]
        @test reproducibility(model, X, y, loss)
    end
    X, y = make_regression(rng=stable_rng());
    loss = LPLoss(p=2)
    for model in [
        DecisionTreeRegressor(),
        RandomForestRegressor(),
    ]
        @test reproducibility(model, X, y, loss)
    end
end

@testset "feature importance defined" begin
    for model ∈ [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostStumpClassifier(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        ]

        @test reports_feature_importances(model) == true
    end
end

@testset "impurity importance" begin

    X, y = MLJBase.make_blobs(100, 3; rng=stable_rng())

    for model ∈ [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostStumpClassifier(),
        ]
        m = machine(model, X, y)
        fit!(m)
        rpt = MLJBase.report(m)
        fi = MLJBase.feature_importances(model, m.fitresult, rpt)
        @test size(fi,1) == 3
    end


    X, y = make_regression(100,3; rng=stable_rng());
    for model in [
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        ]
        m = machine(model, X, y)
        fit!(m)
        rpt = MLJBase.report(m)
        fi = MLJBase.feature_importances(model, m.fitresult, rpt)
        @test size(fi,1) == 3
    end
end

@testset "split importance" begin
    X, y = MLJBase.make_blobs(100, 3; rng=stable_rng())

    for model ∈ [
        DecisionTreeClassifier(feature_importance=:split),
        RandomForestClassifier(feature_importance=:split),
        AdaBoostStumpClassifier(feature_importance=:split),
        ]
        m = machine(model, X, y)
        fit!(m)
        rpt = MLJBase.report(m)
        fi = MLJBase.feature_importances(model, m.fitresult, rpt)
        @test size(fi,1) == 3
    end


    X, y = make_regression(100,3; rng=stable_rng());
    for model in [
        DecisionTreeRegressor(feature_importance=:split),
        RandomForestRegressor(feature_importance=:split),
        ]
        m = machine(model, X, y)
        fit!(m)
        rpt = MLJBase.report(m)
        fi = MLJBase.feature_importances(model, m.fitresult, rpt)
        @test size(fi,1) == 3
    end
end

@testset "warm restart" begin
    for M in [:RandomForestClassifier, :RandomForestRegressor]
        data = (M == :RandomForestClassifier ? make_blobs() : make_regression())
        quote
            model = $M(n_trees=4, rng = stable_rng()) # model with 4 trees
            @test MLJBase.iteration_parameter(model) === :n_trees
            mach =  machine(model, $data...)
            fit!(mach, verbosity=0)
            forest1_4 = fitted_params(mach).forest
            @test length(forest1_4) ==4

            # increase n_trees:
            mach.model = $M(n_trees=7, rng = stable_rng())
            @test_logs(
                (:info, r""),
                (:info, MLJDecisionTreeInterface.info_adding(3)),
                fit!(mach, verbosity=1),
            )

            # decrease n_trees:
            mach.model = $M(n_trees=5, rng = stable_rng())
            @test_logs(
                (:info, r""),
                (:info, MLJDecisionTreeInterface.info_dropping(2)),
                fit!(mach, verbosity=1),
            )
            forest1_5 = fitted_params(mach).forest
            @test length(forest1_5) == 5

            # change a different hyperparameter:
            mach.model = $M(n_trees=5, rng = stable_rng(), max_depth=1)
            @test_logs(
                (:info, r""),
                (:info, MLJDecisionTreeInterface.info_recomputing(5)),
                fit!(mach, verbosity=1),
            )
            forest1_5_again = fitted_params(mach).forest
            @test length(forest1_5_again) == 5
        end |> eval
    end
end

true

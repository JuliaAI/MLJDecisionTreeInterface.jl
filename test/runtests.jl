using Test
import CategoricalArrays
import CategoricalArrays.categorical
using MLJBase
using StableRNGs
using Random
Random.seed!(1234)

stable_rng() = StableRNGs.StableRNG(123)

# load code to be tested:
import DecisionTree
using MLJDecisionTreeInterface

# get some test data:
X, y = @load_iris

baretree = DecisionTreeClassifier(rng=stable_rng())

baretree.max_depth = 1
fitresult, cache, report = MLJBase.fit(baretree, 2, X, y);
baretree.max_depth = -1 # no max depth
fitresult, cache, report =
    MLJBase.update(baretree, 1, fitresult, cache, X, y);

# in this case decision tree is a perfect predictor:
yhat = MLJBase.predict_mode(baretree, fitresult, X);
@test yhat == y

# but pruning upsets this:
baretree.post_prune = true
baretree.merge_purity_threshold=0.1
fitresult, cache, report =
    MLJBase.update(baretree, 2, fitresult, cache, X, y)
yhat = MLJBase.predict_mode(baretree, fitresult, X);
@test yhat != y
yhat = MLJBase.predict(baretree, fitresult, X);

# check preservation of levels:
yyhat = predict_mode(baretree, fitresult, MLJBase.selectrows(X, 1:3))
@test MLJBase.classes(yyhat[1]) == MLJBase.classes(y[1])

# check report and fitresult fields:
@test Set([:classes_seen, :print_tree, :features]) == Set(keys(report))
@test Set(report.classes_seen) == Set(levels(y))
@test report.print_tree(2) === nothing # :-(
@test report.features == [:sepal_length, :sepal_width, :petal_length, :petal_width]

fp = fitted_params(baretree, fitresult)
@test Set([:tree, :encoding, :features]) == Set(keys(fp))
@test fp.features == report.features
enc = fp.encoding
@test Set(values(enc)) == Set(["virginica", "setosa", "versicolor"])
@test enc[MLJBase.int(y[end])] == "virginica"

using Random: seed!
seed!(0)

n,m = 10^3, 5;
raw_features = rand(stable_rng(), n,m);
weights = rand(stable_rng(), -1:1,m);
labels = raw_features * weights;
features = MLJBase.table(raw_features);

R1Tree = DecisionTreeRegressor(
    min_samples_leaf=5,
    merge_purity_threshold=0.1,
    rng=stable_rng(),
)
R2Tree = DecisionTreeRegressor(min_samples_split=5, rng=stable_rng())
model1, = MLJBase.fit(R1Tree,1, features, labels)

vals1 = MLJBase.predict(R1Tree,model1,features)
R1Tree.post_prune = true
model1_prune, = MLJBase.fit(R1Tree,1, features, labels)
vals1_prune = MLJBase.predict(R1Tree,model1_prune,features)
@test vals1 !=vals1_prune

@test DecisionTree.R2(labels, vals1) > 0.8

model2, = MLJBase.fit(R2Tree, 1, features, labels)
vals2 = MLJBase.predict(R2Tree, model2, features)
@test DecisionTree.R2(labels, vals2) > 0.8


## TEST ON ORDINAL FEATURES OTHER THAN CONTINUOUS

N = 20
X = (
    x1=rand(stable_rng(),N),
    x2=categorical(rand(stable_rng(), "abc", N), ordered=true),
    x3=collect(1:N),
)
yfinite = X.x2
ycont = float.(X.x3)

rgs = DecisionTreeRegressor(rng=stable_rng())
fitresult, _, _ = MLJBase.fit(rgs, 1, X, ycont)
@test rms(predict(rgs, fitresult, X), ycont) < 1.5

clf = DecisionTreeClassifier()
fitresult, _, _ = MLJBase.fit(clf, 1, X, yfinite)
@test sum(predict(clf, fitresult, X) .== yfinite) == 0 # perfect prediction


# --  Ensemble

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



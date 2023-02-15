module MLJDecisionTreeInterface

import MLJModelInterface
using MLJModelInterface.ScientificTypesBase
import DecisionTree
import Tables
using CategoricalArrays

using Random
import Random.GLOBAL_RNG

const MMI = MLJModelInterface
const DT = DecisionTree
const PKG = "MLJDecisionTreeInterface"

struct TreePrinter{T}
    tree::T
end
(c::TreePrinter)(depth) = DT.print_tree(c.tree, depth)
(c::TreePrinter)() = DT.print_tree(c.tree, 5)

Base.show(stream::IO, c::TreePrinter) =
    print(stream, "TreePrinter object (call with display depth)")

function classes(y)
    p = CategoricalArrays.pool(y)
    [p[i] for i in 1:length(p)]
end

# # DECISION TREE CLASSIFIER

MMI.@mlj_model mutable struct DecisionTreeClassifier <: MMI.Probabilistic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = 0::(_ ≥ -1)
    post_prune::Bool             = false
    merge_purity_threshold::Float64 = 1.0::(_ ≤ 1)
    display_depth::Int           = 5::(_ ≥ 1)
    feature_importance::Symbol = :impurity::(_ ∈ (:impurity, :split))
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(
    m::DecisionTreeClassifier,
    verbosity::Int,
    Xmatrix,
    yplain,
    features,
    classes,
    )

    integers_seen = unique(yplain)
    classes_seen  = MMI.decoder(classes)(integers_seen)

    tree = DT.build_tree(yplain, Xmatrix,
                         m.n_subfeatures,
                         m.max_depth,
                         m.min_samples_leaf,
                         m.min_samples_split,
                         m.min_purity_increase,
                         rng=m.rng)
    if m.post_prune
        tree = DT.prune_tree(tree, m.merge_purity_threshold)
    end
    verbosity < 2 || DT.print_tree(tree, m.display_depth)

    fitresult = (tree, classes_seen, integers_seen, features)

    cache  = nothing
    report = (
        classes_seen=classes_seen,
        print_tree=TreePrinter(tree),
        features=features,
    )
    return fitresult, cache, report
end

get_encoding(classes_seen) = Dict(MMI.int(c) => c for c in classes(classes_seen))

MMI.fitted_params(::DecisionTreeClassifier, fitresult) =
    (tree=fitresult[1],
     encoding=get_encoding(fitresult[2]),
     features=fitresult[4])

function MMI.predict(m::DecisionTreeClassifier, fitresult, Xnew)
    tree, classes_seen, integers_seen = fitresult
    # retrieve the predicted scores
    scores = DT.apply_tree_proba(tree, Xnew, integers_seen)
    MMI.UnivariateFinite(classes_seen, scores)
end

MMI.reports_feature_importances(::Type{<:DecisionTreeClassifier}) = true


# # RANDOM FOREST CLASSIFIER

MMI.@mlj_model mutable struct RandomForestClassifier <: MMI.Probabilistic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
    n_trees::Int                 = 10::(_ ≥ 2)
    sampling_fraction::Float64   = 0.7::(0 < _ ≤ 1)
    feature_importance::Symbol = :impurity::(_ ∈ (:impurity, :split))
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(
    m::RandomForestClassifier,
    verbosity::Int,
    Xmatrix,
    yplain,
    features,
    classes,
    )

    integers_seen = unique(yplain)
    classes_seen  = MMI.decoder(classes)(integers_seen)

    forest = DT.build_forest(yplain, Xmatrix,
                             m.n_subfeatures,
                             m.n_trees,
                             m.sampling_fraction,
                             m.max_depth,
                             m.min_samples_leaf,
                             m.min_samples_split,
                             m.min_purity_increase;
                             rng=m.rng)
    cache  = nothing

    report = (features=features,)

    return (forest, classes_seen, integers_seen), cache, report
end

MMI.fitted_params(::RandomForestClassifier, (forest,_)) = (forest=forest,)

function MMI.predict(m::RandomForestClassifier, fitresult, Xnew)
    forest, classes_seen, integers_seen = fitresult
    scores = DT.apply_forest_proba(forest, Xnew, integers_seen)
    MMI.UnivariateFinite(classes_seen, scores)
end

MMI.reports_feature_importances(::Type{<:RandomForestClassifier}) = true


# # ADA BOOST STUMP CLASSIFIER

MMI.@mlj_model mutable struct AdaBoostStumpClassifier <: MMI.Probabilistic
    n_iter::Int            = 10::(_ ≥ 1)
    feature_importance::Symbol = :impurity::(_ ∈ (:impurity, :split))
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(
    m::AdaBoostStumpClassifier,
    verbosity::Int,
    Xmatrix,
    yplain,
    features,
    classes,
    )

    integers_seen = unique(yplain)
    classes_seen  = MMI.decoder(classes)(integers_seen)

    stumps, coefs =
        DT.build_adaboost_stumps(yplain, Xmatrix, m.n_iter, rng=m.rng)
    cache  = nothing

    report = (features=features,)

    return (stumps, coefs, classes_seen, integers_seen), cache, report
end

MMI.fitted_params(::AdaBoostStumpClassifier, (stumps,coefs,_)) =
    (stumps=stumps,coefs=coefs)

function MMI.predict(m::AdaBoostStumpClassifier, fitresult, Xnew)
    stumps, coefs, classes_seen, integers_seen = fitresult
    scores = DT.apply_adaboost_stumps_proba(
        stumps,
        coefs,
        Xnew,
        integers_seen,
    )
    return MMI.UnivariateFinite(classes_seen, scores)
end

MMI.reports_feature_importances(::Type{<:AdaBoostStumpClassifier}) = true


# # DECISION TREE REGRESSOR

MMI.@mlj_model mutable struct DecisionTreeRegressor <: MMI.Deterministic
    max_depth::Int                               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int                = 5::(_ ≥ 0)
    min_samples_split::Int               = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int                   = 0::(_ ≥ -1)
    post_prune::Bool                     = false
    merge_purity_threshold::Float64 = 1.0::(0 ≤ _ ≤ 1)
    feature_importance::Symbol = :impurity::(_ ∈ (:impurity, :split))
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(m::DecisionTreeRegressor, verbosity::Int, Xmatrix, y, features)

    tree = DT.build_tree(
        y,
        Xmatrix,
        m.n_subfeatures,
        m.max_depth,
        m.min_samples_leaf,
        m.min_samples_split,
        m.min_purity_increase;
        rng=m.rng
    )

    if m.post_prune
        tree = DT.prune_tree(tree, m.merge_purity_threshold)
    end
    cache  = nothing

    report = (features=features,)

    return tree, cache, report
end

MMI.fitted_params(::DecisionTreeRegressor, tree) = (tree=tree,)

MMI.predict(::DecisionTreeRegressor, tree, Xnew) = DT.apply_tree(tree, Xnew)

MMI.reports_feature_importances(::Type{<:DecisionTreeRegressor}) = true


# # RANDOM FOREST REGRESSOR

MMI.@mlj_model mutable struct RandomForestRegressor <: MMI.Deterministic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
    n_trees::Int                 = 10::(_ ≥ 2)
    sampling_fraction::Float64   = 0.7::(0 < _ ≤ 1)
    feature_importance::Symbol = :impurity::(_ ∈ (:impurity, :split))
    rng::Union{AbstractRNG,Integer} = GLOBAL_RNG
end

function MMI.fit(m::RandomForestRegressor, verbosity::Int, Xmatrix, y, features)

    forest = DT.build_forest(
        y,
        Xmatrix,
        m.n_subfeatures,
        m.n_trees,
        m.sampling_fraction,
        m.max_depth,
        m.min_samples_leaf,
        m.min_samples_split,
        m.min_purity_increase,
        rng=m.rng
    )

    cache  = nothing
    report = (features=features,)

    return forest, cache, report
end

MMI.fitted_params(::RandomForestRegressor, forest) = (forest=forest,)

MMI.predict(::RandomForestRegressor, forest, Xnew) = DT.apply_forest(forest, Xnew)

MMI.reports_feature_importances(::Type{<:RandomForestRegressor}) = true

# # ALIASES FOR TYPE UNIONS

const TreeModel = Union{
    DecisionTreeClassifier,
    RandomForestClassifier,
    AdaBoostStumpClassifier,
    DecisionTreeRegressor,
    RandomForestRegressor,
}

const IterativeModel = Union{
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostStumpClassifier,
}

const Classifier = Union{
    DecisionTreeClassifier,
    RandomForestClassifier,
    AdaBoostStumpClassifier,
}

const Regressor = Union{
    DecisionTreeRegressor,
    RandomForestRegressor,
}

const RandomForestModel = Union{
    DecisionTreeClassifier,
    RandomForestClassifier,
}


# # DATA FRONT END

_columnnames(X) = Tables.columnnames(Tables.columns(X)) |> collect

# for fit:
MMI.reformat(::Classifier, X, y) =
    (Tables.matrix(X), MMI.int(y), _columnnames(X), classes(y))
MMI.reformat(::Regressor, X, y) =
    (Tables.matrix(X), float(y), _columnnames(X))
MMI.selectrows(::TreeModel, I, Xmatrix, y, meta...) =
    (view(Xmatrix, I, :), view(y, I), meta...)

# for predict:
MMI.reformat(::TreeModel, X) = (Tables.matrix(X),)
MMI.selectrows(::TreeModel, I, Xmatrix) = (view(Xmatrix, I, :),)


# # FEATURE IMPORTANCES

# get actual arguments needed for importance calculation from various fitresults.
get_fitresult(
    m::Union{DecisionTreeClassifier, RandomForestClassifier},
    fitresult,
) = (fitresult[1],)
get_fitresult(
    m::Union{DecisionTreeRegressor, RandomForestRegressor},
    fitresult,
) = (fitresult,)
get_fitresult(m::AdaBoostStumpClassifier, fitresult)= (fitresult[1], fitresult[2])

function MMI.feature_importances(m::TreeModel, fitresult, report)
    # generate feature importances for report
    if m.feature_importance == :impurity
        feature_importance_func = DT.impurity_importance
    elseif m.feature_importance == :split
        feature_importance_func = DT.split_importance
    end

    mdl = get_fitresult(m, fitresult)
    features = report.features
    fi = feature_importance_func(mdl..., normalize=true)
    fi_pairs = Pair.(features, fi)
    # sort descending
    sort!(fi_pairs, by= x->-x[2])

    return fi_pairs
end

# # METADATA (MODEL TRAITS)

MMI.metadata_pkg.(
    (DecisionTreeClassifier, DecisionTreeRegressor,
     RandomForestClassifier, RandomForestRegressor,
     AdaBoostStumpClassifier),
    name = "DecisionTree",
    package_uuid = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb",
    package_url = "https://github.com/bensadeghi/DecisionTree.jl",
    is_pure_julia = true,
    package_license = "MIT"
)

MMI.metadata_model(
    DecisionTreeClassifier,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{<:Finite},
    human_name = "CART decision tree classifier",
    load_path = "$PKG.DecisionTreeClassifier"
)

MMI.metadata_model(
    RandomForestClassifier,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{<:Finite},
    human_name = "CART random forest classifier",
    load_path = "$PKG.RandomForestClassifier"
)

MMI.metadata_model(
    AdaBoostStumpClassifier,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{<:Finite},
    human_name = "Ada-boosted stump classifier",
    load_path = "$PKG.AdaBoostStumpClassifier"
)

MMI.metadata_model(
    DecisionTreeRegressor,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{Continuous},
    human_name = "CART decision tree regressor",
    load_path = "$PKG.DecisionTreeRegressor"
)

MMI.metadata_model(
    RandomForestRegressor,
    input_scitype = Table(Continuous, Count, OrderedFactor),
    target_scitype = AbstractVector{Continuous},
    human_name = "CART random forest regressor",
    load_path = "$PKG.RandomForestRegressor")


# # DOCUMENT STRINGS

const DOC_CART = "[CART algorithm](https://en.wikipedia.org/wiki/Decision_tree_learning)"*
", originally published in Breiman, Leo; Friedman, J. H.; Olshen, R. A.; "*
"Stone, C. J. (1984): \"Classification and regression trees\". *Monterey, "*
"CA: Wadsworth & Brooks/Cole Advanced Books & Software.*"

const DOC_RANDOM_FOREST = "[Random Forest algorithm]"*
    "(https://en.wikipedia.org/wiki/Random_forest), originally published in "*
    "Breiman, L. (2001): \"Random Forests.\", *Machine Learning*, vol. 45, pp. 5–32"

"""
$(MMI.doc_header(DecisionTreeClassifier))

`DecisionTreeClassifier` implements the $DOC_CART.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`

- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `max_depth=-1`:          max depth of the decision tree (-1=any)

- `min_samples_leaf=1`:    max number of samples each leaf needs to have

- `min_samples_split=2`:   min number of samples needed for a split

- `min_purity_increase=0`: min purity needed for a split

- `n_subfeatures=0`: number of features to select at random (0 for all)

- `post_prune=false`:      set to `true` for post-fit pruning

- `merge_purity_threshold=1.0`: (post-pruning) merge leaves having
                           combined purity `>= merge_purity_threshold`

- `display_depth=5`:       max depth to show when displaying the tree

- `feature_importance`: method to use for computing feature importances. One of `(:impurity,
  :split)`

- `rng=Random.GLOBAL_RNG`: random number generator or seed


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.

- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `tree`: the tree or stump object returned by the core DecisionTree.jl algorithm

- `encoding`: dictionary of target classes keyed on integers used
  internally by DecisionTree.jl; needed to interpret pretty printing
  of tree (obtained by calling `fit!(mach, verbosity=2)` or from
  report - see below)

- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)


# Report

The fields of `report(mach)` are:

- `classes_seen`: list of target classes actually observed in training

- `print_tree`: method to print a pretty representation of the fitted
  tree, with single argument the tree depth; interpretation requires
  internal integer-class encoding (see "Fitted parameters" above).

- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)


# Examples

```
using MLJ
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree(max_depth=4, min_samples_split=3)

X, y = @load_iris
mach = machine(tree, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
pdf.(yhat, "virginica")    # probabilities for the "verginica" class

fitted_params(mach).tree # raw tree or stump object from DecisionTrees.jl

julia> report(mach).print_tree(3)
Feature 4, Threshold 0.8
L-> 1 : 50/50
R-> Feature 4, Threshold 1.75
    L-> Feature 3, Threshold 4.95
        L->
        R->
    R-> Feature 3, Threshold 4.85
        L->
        R-> 3 : 43/43
```

To interpret the internal class labelling:

```
julia> fitted_params(mach).encoding
Dict{CategoricalArrays.CategoricalValue{String, UInt32}, UInt32} with 3 entries:
  0x00000003 => "virginica"
  0x00000001 => "setosa"
  0x00000002 => "versicolor"
```

See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type [`MLJDecisionTreeInterface.DecisionTree.DecisionTreeClassifier`](@ref).

"""
DecisionTreeClassifier

"""
$(MMI.doc_header(RandomForestClassifier))

`RandomForestClassifier` implements the standard $DOC_RANDOM_FOREST.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.


# Hyper-parameters

- `max_depth=-1`:          max depth of the decision tree (-1=any)

- `min_samples_leaf=1`:    min number of samples each leaf needs to have

- `min_samples_split=2`:   min number of samples needed for a split

- `min_purity_increase=0`: min purity needed for a split

- `n_subfeatures=-1`: number of features to select at random (0 for all,
  -1 for square root of number of features)

- `n_trees=10`:            number of trees to train

- `sampling_fraction=0.7`  fraction of samples to train each tree on

- `feature_importance`: method to use for computing feature importances. One of `(:impurity,
  :split)`

- `rng=Random.GLOBAL_RNG`: random number generator or seed


# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.

- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `forest`: the `Ensemble` object returned by the core DecisionTree.jl algorithm


# Report

- `features`: the names of the features encountered in training


# Examples

```
using MLJ
Forest = @load RandomForestClassifier pkg=DecisionTree
forest = Forest(min_samples_split=6, n_subfeatures=3)

X, y = @load_iris
mach = machine(forest, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
pdf.(yhat, "virginica")    # probabilities for the "verginica" class

fitted_params(mach).forest # raw `Ensemble` object from DecisionTrees.jl

feature_importances(mach)  # `:impurity` feature importances
forest.feature_importance = :split
feature_importance(mach)   # `:split` feature importances

```
See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.RandomForestClassifier`](@ref).

"""
RandomForestClassifier

"""
$(MMI.doc_header(AdaBoostStumpClassifier))


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where:

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.


# Hyper-parameters

- `n_iter=10`:   number of iterations of AdaBoost

- `feature_importance`: method to use for computing feature importances. One of `(:impurity,
  :split)`

- `rng=Random.GLOBAL_RNG`: random number generator or seed

# Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are probabilistic, but uncalibrated.

- `predict_mode(mach, Xnew)`: instead return the mode of each
  prediction above.


# Fitted Parameters

The fields of `fitted_params(mach)` are:

- `stumps`: the `Ensemble` object returned by the core DecisionTree.jl
  algorithm.

- `coefficients`: the stump coefficients (one per stump)


# Report

- `features`: the names of the features encountered in training


```
using MLJ
Booster = @load AdaBoostStumpClassifier pkg=DecisionTree
booster = Booster(n_iter=15)

X, y = @load_iris
mach = machine(booster, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
pdf.(yhat, "virginica")    # probabilities for the "verginica" class

fitted_params(mach).stumps # raw `Ensemble` object from DecisionTree.jl
fitted_params(mach).coefs  # coefficient associated with each stump
```

See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.AdaBoostStumpClassifier`](@ref).

"""
AdaBoostStumpClassifier

"""
$(MMI.doc_header(DecisionTreeRegressor))

`DecisionTreeRegressor` implements the $DOC_CART.


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.


# Hyper-parameters

- `max_depth=-1`:          max depth of the decision tree (-1=any)

- `min_samples_leaf=1`:    max number of samples each leaf needs to have

- `min_samples_split=2`:   min number of samples needed for a split

- `min_purity_increase=0`: min purity needed for a split

- `n_subfeatures=0`: number of features to select at random (0 for all)

- `post_prune=false`:      set to `true` for post-fit pruning

- `merge_purity_threshold=1.0`: (post-pruning) merge leaves having
                           combined purity `>= merge_purity_threshold`

- `feature_importance`:    method to use for computing feature importances. One of `(:impurity, :split)`

- `rng=Random.GLOBAL_RNG`: random number generator or seed


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `tree`: the tree or stump object returned by the core
  DecisionTree.jl algorithm


# Report

- `features`: the names of the features encountered in training


# Examples

```
using MLJ
Tree = @load DecisionTreeRegressor pkg=DecisionTree
tree = Tree(max_depth=4, min_samples_split=3)

X, y = make_regression(100, 2) # synthetic data
mach = machine(tree, X, y) |> fit!

Xnew, _ = make_regression(3, 2)
yhat = predict(mach, Xnew) # new predictions

fitted_params(mach).tree # raw tree or stump object from DecisionTree.jl
```

See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.DecisionTreeRegressor`](@ref).

"""
DecisionTreeRegressor

"""
$(MMI.doc_header(RandomForestRegressor))

`DecisionTreeRegressor` implements the standard $DOC_RANDOM_FOREST


# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `Continuous`; check the scitype with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.


# Hyper-parameters

- `max_depth=-1`:          max depth of the decision tree (-1=any)

- `min_samples_leaf=1`:    min number of samples each leaf needs to have

- `min_samples_split=2`:   min number of samples needed for a split

- `min_purity_increase=0`: min purity needed for a split

- `n_subfeatures=-1`: number of features to select at random (0 for all,
  -1 for square root of number of features)

- `n_trees=10`:            number of trees to train

- `sampling_fraction=0.7`  fraction of samples to train each tree on

- `feature_importance`:    method to use for computing feature importances. One of `(:impurity, :split)`

- `rng=Random.GLOBAL_RNG`: random number generator or seed


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `forest`: the `Ensemble` object returned by the core DecisionTree.jl algorithm


# Report

- `features`: the names of the features encountered in training


# Examples

```
using MLJ
Forest = @load RandomForestRegressor pkg=DecisionTree
forest = Forest(max_depth=4, min_samples_split=3)

X, y = make_regression(100, 2) # synthetic data
mach = machine(forest, X, y) |> fit!

Xnew, _ = make_regression(3, 2)
yhat = predict(mach, Xnew) # new predictions

fitted_params(mach).forest # raw `Ensemble` object from DecisionTree.jl
```

See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.RandomForestRegressor`](@ref).

"""
RandomForestRegressor

end # module

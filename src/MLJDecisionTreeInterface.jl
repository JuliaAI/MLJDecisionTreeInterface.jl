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
    features::Vector{Symbol}
end
(c::TreePrinter)(depth) = DT.print_tree(c.tree, depth, feature_names = c.features)
(c::TreePrinter)() = DT.print_tree(c.tree, 5, feature_names = c.features)

Base.show(stream::IO, c::TreePrinter) =
    print(stream, "TreePrinter object (call with display depth)")


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
        print_tree=TreePrinter(tree, features),
        features=features,
    )
    return fitresult, cache, report
end

# returns a dictionary of categorical elements keyed on ref integer:
get_encoding(classes_seen) = Dict(MMI.int(c) => c for c in levels(classes_seen))

# given such a dictionary, return printable class labels, ordered by corresponding ref
# integer:
classlabels(encoding) = [string(encoding[i]) for i in sort(keys(encoding) |> collect)]

_node_or_leaf(r::DecisionTree.Root) = r.node
_node_or_leaf(n::Any) = n

function MMI.fitted_params(::DecisionTreeClassifier, fitresult)
    raw_tree = fitresult[1]
    encoding = get_encoding(fitresult[2])
    features = fitresult[4]
    classlabels = MLJDecisionTreeInterface.classlabels(encoding)
    tree = DecisionTree.wrap(
        _node_or_leaf(raw_tree),
        (featurenames=features, classlabels),
    )
    (; tree, raw_tree, encoding, features)
end

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
    n_trees::Int                 = 100::(_ ≥ 0)
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
    cache  = deepcopy(m)

    report = (features=features,)

    return (forest, classes_seen, integers_seen), cache, report
end

info_recomputing(n) = "Detected a change to hyperparameters " *
    "not restricted to `n_trees`. Recomputing all $n trees trees. "
info_dropping(n) =  "Dropping $n trees from the ensemble. "
info_adding(n) = "Adding $n trees to the ensemble. "

function MMI.update(
    model::RandomForestClassifier,
    verbosity::Int,
    old_fitresult,
    old_model,
    Xmatrix,
    yplain,
    features,
    classes,
    )

    only_iterations_have_changed = MMI.is_same_except(model, old_model, :n_trees)

    if !only_iterations_have_changed
        verbosity > 0 && @info info_recomputing(model.n_trees)
        return MMI.fit(
            model,
            verbosity,
            Xmatrix,
            yplain,
            features,
            classes,
        )
    end

    old_forest = old_fitresult[1]
    Δn_trees = model.n_trees - old_model.n_trees
    # if `n_trees` drops, then tuncate, otherwise compute more trees
    if Δn_trees < 0
        verbosity > 0 && @info info_dropping(-Δn_trees)
        forest = old_forest[1:model.n_trees]
    else
        verbosity > 0 && @info info_adding(Δn_trees)
        forest = DT.build_forest(
            old_forest,
            yplain, Xmatrix,
            model.n_subfeatures,
            model.n_trees,
            model.sampling_fraction,
            model.max_depth,
            model.min_samples_leaf,
            model.min_samples_split,
            model.min_purity_increase;
            rng=model.rng,
        )
    end

    fitresult = (forest, old_fitresult[2:3]...)
    cache = deepcopy(model)
    report = (features=features,)
    return fitresult, cache, report

end

MMI.fitted_params(::RandomForestClassifier, (forest,_)) = (forest=forest,)

function MMI.predict(m::RandomForestClassifier, fitresult, Xnew)
    forest, classes_seen, integers_seen = fitresult
    scores = DT.apply_forest_proba(forest, Xnew, integers_seen)
    MMI.UnivariateFinite(classes_seen, scores)
end

MMI.reports_feature_importances(::Type{<:RandomForestClassifier}) = true
MMI.iteration_parameter(::Type{<:RandomForestClassifier}) = :n_trees

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
    fitresult = (tree, features)

    return fitresult, cache, report
end

function MMI.fitted_params(::DecisionTreeRegressor, fitresult)
    raw_tree = fitresult[1]
    features = fitresult[2]
    tree = DecisionTree.wrap(
        _node_or_leaf(raw_tree),
        (; featurenames=features),
    )
    (; tree, raw_tree)
end

MMI.predict(::DecisionTreeRegressor, fitresult, Xnew) = DT.apply_tree(fitresult[1], Xnew)

MMI.reports_feature_importances(::Type{<:DecisionTreeRegressor}) = true


# # RANDOM FOREST REGRESSOR

MMI.@mlj_model mutable struct RandomForestRegressor <: MMI.Deterministic
    max_depth::Int               = (-)(1)::(_ ≥ -1)
    min_samples_leaf::Int        = 1::(_ ≥ 0)
    min_samples_split::Int       = 2::(_ ≥ 2)
    min_purity_increase::Float64 = 0.0::(_ ≥ 0)
    n_subfeatures::Int           = (-)(1)::(_ ≥ -1)
    n_trees::Int                 = 100::(_ ≥ 0)
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

    cache  = deepcopy(m)
    report = (features=features,)

    return forest, cache, report
end

function MMI.update(
    model::RandomForestRegressor,
    verbosity::Int,
    old_forest,
    old_model,
    Xmatrix,
    y,
    features,
    )

    only_iterations_have_changed = MMI.is_same_except(model, old_model, :n_trees)

    if !only_iterations_have_changed
        verbosity > 0 && @info info_recomputing(model.n_trees)
        return MMI.fit(
            model,
            verbosity,
            Xmatrix,
            y,
            features,
        )
    end

    Δn_trees = model.n_trees - old_model.n_trees

    # if `n_trees` drops, then tuncate, otherwise compute more trees
    if Δn_trees < 0
        verbosity > 0 && @info info_dropping(-Δn_trees)
        forest = old_forest[1:model.n_trees]
    else
        verbosity > 0 && @info info_adding(Δn_trees)
        forest = DT.build_forest(
            old_forest,
            y,
            Xmatrix,
            model.n_subfeatures,
            model.n_trees,
            model.sampling_fraction,
            model.max_depth,
            model.min_samples_leaf,
            model.min_samples_split,
            model.min_purity_increase;
            rng=model.rng
        )
    end

    cache = deepcopy(model)
    report = (features=features,)

    return forest, cache, report

end

MMI.fitted_params(::RandomForestRegressor, forest) = (forest=forest,)

MMI.predict(::RandomForestRegressor, forest, Xnew) = DT.apply_forest(forest, Xnew)

MMI.reports_feature_importances(::Type{<:RandomForestRegressor}) = true
MMI.iteration_parameter(::Type{<:RandomForestRegressor}) = :n_trees


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

# to get column names based on table access type:
_columnnames(X) = _columnnames(X, Val(Tables.columnaccess(X))) |> collect
_columnnames(X, ::Val{true}) = Tables.columnnames(Tables.columns(X))
_columnnames(X, ::Val{false}) = Tables.columnnames(first(Tables.rows(X)))

# for fit:
MMI.reformat(::Classifier, X, y) =
    (Tables.matrix(X), MMI.int(y), _columnnames(X), levels(y))
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
    m::Union{DecisionTreeClassifier, RandomForestClassifier, DecisionTreeRegressor},
    fitresult,
) = (fitresult[1],)
get_fitresult(
    m::RandomForestRegressor,
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


# Hyperparameters

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

- `raw_tree`: the raw `Node`, `Leaf` or `Root` object returned by the core DecisionTree.jl
  algorithm

- `tree`: a visualizable, wrapped version of `raw_tree` implementing the AbstractTrees.jl
  interface; see "Examples" below

- `encoding`: dictionary of target classes keyed on integers used
  internally by DecisionTree.jl

- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)


# Report

The fields of `report(mach)` are:

- `classes_seen`: list of target classes actually observed in training

- `print_tree`: alternative method to print the fitted
  tree, with single argument the tree depth; interpretation requires
  internal integer-class encoding (see "Fitted parameters" above).

- `features`: the names of the features encountered in training, in an
  order consistent with the output of `print_tree` (see below)

# Accessor functions

- `feature_importances(mach)` returns a vector of `(feature::Symbol => importance)` pairs;
  the type of importance is determined by the hyperparameter `feature_importance` (see
  above)

# Examples

```
using MLJ
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model = DecisionTreeClassifier(max_depth=3, min_samples_split=3)

X, y = @load_iris
mach = machine(model, X, y) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)
yhat = predict(mach, Xnew) # probabilistic predictions
predict_mode(mach, Xnew)   # point predictions
pdf.(yhat, "virginica")    # probabilities for the "verginica" class

julia> tree = fitted_params(mach).tree
petal_length < 2.45
├─ setosa (50/50)
└─ petal_width < 1.75
   ├─ petal_length < 4.95
   │  ├─ versicolor (47/48)
   │  └─ virginica (4/6)
   └─ petal_length < 4.85
      ├─ virginica (2/3)
      └─ virginica (43/43)

using Plots, TreeRecipe
plot(tree) # for a graphical representation of the tree

feature_importances(mach)
```

See also [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and the
unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.DecisionTreeClassifier`](@ref).

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


# Hyperparameters

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

The fields of `report(mach)` are:

- `features`: the names of the features encountered in training


# Accessor functions

- `feature_importances(mach)` returns a vector of `(feature::Symbol => importance)` pairs;
  the type of importance is determined by the hyperparameter `feature_importance` (see
  above)


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


# Hyperparameters

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

The fields of `report(mach)` are:

- `features`: the names of the features encountered in training


# Accessor functions

- `feature_importances(mach)` returns a vector of `(feature::Symbol => importance)` pairs;
  the type of importance is determined by the hyperparameter `feature_importance` (see
  above)


# Examples

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
feature_importances(mach)
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


# Hyperparameters

- `max_depth=-1`:          max depth of the decision tree (-1=any)

- `min_samples_leaf=1`:    max number of samples each leaf needs to have

- `min_samples_split=2`:   min number of samples needed for a split

- `min_purity_increase=0`: min purity needed for a split

- `n_subfeatures=0`: number of features to select at random (0 for all)

- `post_prune=false`:      set to `true` for post-fit pruning

- `merge_purity_threshold=1.0`: (post-pruning) merge leaves having
                           combined purity `>= merge_purity_threshold`

- `feature_importance`: method to use for computing feature importances. One of
  `(:impurity, :split)`

- `rng=Random.GLOBAL_RNG`: random number generator or seed


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `tree`: the tree or stump object returned by the core
  DecisionTree.jl algorithm

- `features`: the names of the features encountered in training


# Report

The fields of `report(mach)` are:

- `features`: the names of the features encountered in training


# Accessor functions

- `feature_importances(mach)` returns a vector of `(feature::Symbol => importance)` pairs;
  the type of importance is determined by the hyperparameter `feature_importance` (see
  above)


# Examples

```
using MLJ
DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
model = DecisionTreeRegressor(max_depth=3, min_samples_split=3)

X, y = make_regression(100, 4; rng=123) # synthetic data
mach = machine(model, X, y) |> fit!

Xnew, _ = make_regression(3, 2; rng=123)
yhat = predict(mach, Xnew) # new predictions

julia> fitted_params(mach).tree
x1 < 0.2758
├─ x2 < 0.9137
│  ├─ x1 < -0.9582
│  │  ├─ 0.9189256882087312 (0/12)
│  │  └─ -0.23180616021065256 (0/38)
│  └─ -1.6461153800037722 (0/9)
└─ x1 < 1.062
   ├─ x2 < -0.4969
   │  ├─ -0.9330755147107384 (0/5)
   │  └─ -2.3287967825015548 (0/17)
   └─ x2 < 0.4598
      ├─ -2.931299926506291 (0/11)
      └─ -4.726518740473489 (0/8)

feature_importances(mach) # get feature importances
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


# Hyperparameters

- `max_depth=-1`: max depth of the decision tree (-1=any)

- `min_samples_leaf=1`: min number of samples each leaf needs to have

- `min_samples_split=2`: min number of samples needed for a split

- `min_purity_increase=0`: min purity needed for a split

- `n_subfeatures=-1`: number of features to select at random (0 for all,
  -1 for square root of number of features)

- `n_trees=10`: number of trees to train

- `sampling_fraction=0.7`  fraction of samples to train each tree on

- `feature_importance`: method to use for computing feature importances. One of
  `(:impurity, :split)`

- `rng=Random.GLOBAL_RNG`: random number generator or seed


# Operations

- `predict(mach, Xnew)`: return predictions of the target given new
  features `Xnew` having the same scitype as `X` above.


# Fitted parameters

The fields of `fitted_params(mach)` are:

- `forest`: the `Ensemble` object returned by the core DecisionTree.jl algorithm


# Report

The fields of `report(mach)` are:

- `features`: the names of the features encountered in training


# Accessor functions

- `feature_importances(mach)` returns a vector of `(feature::Symbol => importance)` pairs;
  the type of importance is determined by the hyperparameter `feature_importance` (see
  above)


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
feature_importances(mach)
```

See also
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) and
the unwrapped model type
[`MLJDecisionTreeInterface.DecisionTree.RandomForestRegressor`](@ref).

"""
RandomForestRegressor

end # module

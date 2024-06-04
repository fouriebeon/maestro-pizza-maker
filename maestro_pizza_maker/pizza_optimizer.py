# The maestro pizza maker is aware that pizza creation is a complex process and it is not always possible to create a pizza that satisfies all the constraints.
# Therefore maestro wants to create a pizza that will satisfy as many constraints as possible to avoid the risk creating disappointing pizzas like pizza Hawaii.
# To do so, maestro wants to use the optimization techniques.

# TODO: implement the pizza optimizer that will create a pizza that will satisfy all the constraints and will maximize following objective functions:
# obj = Expected_value(taste(pizza)) - lambda * price(pizza), where lambda is a parameter that will be provided by the maestro pizza maker
# hint: use the mip library and the following documentation: https://www.python-mip.com/
# hint: you can find inspiration in the minimize_price function


from dataclasses import dataclass

import numpy as np

from mip import Model, xsum, minimize, maximize, INTEGER, OptimizationStatus

from maestro_pizza_maker.ingredients import IngredientType, PizzaIngredients
from maestro_pizza_maker.pizza import Pizza


@dataclass
class ValueBounds:
    min: float = 0.0
    max: float = np.inf


@dataclass
class PizzaConstraintsValues:
    price: ValueBounds = ValueBounds()
    protein: ValueBounds = ValueBounds()
    fat: ValueBounds = ValueBounds()
    carbohydrates: ValueBounds = ValueBounds()
    calories: ValueBounds = ValueBounds()


@dataclass
class PizzaConstraintsIngredients:
    cheese: int = 0
    fruits: int = 0
    meat: int = 0
    vegetables: int = 0
    dough: int = 1
    sauce: int = 1


def minimize_price(
    constraints_values: PizzaConstraintsValues,
    constraints_ingredients: PizzaConstraintsIngredients,
) -> Pizza:
    """"""
    model = Model()

    # sets
    ingredients = [ingredient for ingredient in PizzaIngredients]
    ingredients_names = [ingredient.name for ingredient in ingredients]

    # variables
    x = [
        model.add_var(var_type=INTEGER, lb=0, ub=1, name=ingredient)
        for ingredient in ingredients_names
    ]

    # objective function
    model.objective = minimize(
        xsum(x[i] * ingredients[i].value.price for i in range(len(ingredients)))
    )

    # constraints
    model += (
        xsum(x[i] * ingredients[i].value.protein for i in range(len(ingredients)))
        >= constraints_values.protein.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.protein for i in range(len(ingredients)))
        <= constraints_values.protein.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.fat.mean() for i in range(len(ingredients)))
        >= constraints_values.fat.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.fat.mean() for i in range(len(ingredients)))
        <= constraints_values.fat.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.carbohydrates for i in range(len(ingredients)))
        >= constraints_values.carbohydrates.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.carbohydrates for i in range(len(ingredients)))
        <= constraints_values.carbohydrates.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.calories for i in range(len(ingredients)))
        >= constraints_values.calories.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.calories for i in range(len(ingredients)))
        <= constraints_values.calories.max
    )

    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.DOUGH
        )
        == constraints_ingredients.dough
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.SAUCE
        )
        == constraints_ingredients.sauce
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.CHEESE
        )
        == constraints_ingredients.cheese
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.MEAT
        )
        == constraints_ingredients.meat
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.VEGETABLE
        )
        == constraints_ingredients.vegetables
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.FRUIT
        )
        == constraints_ingredients.fruits
    )

    # optimize
    model.optimize()

    # check solution
    if model.status != OptimizationStatus.OPTIMAL:
        raise Exception(
            "The model is not optimal -> likely no solution found (infeasible))"
        )

    # solution
    return Pizza(
        dough=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.DOUGH and x[i].x == 1
        ][0],
        sauce=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.SAUCE and x[i].x == 1
        ],
        cheese=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.CHEESE and x[i].x == 1
        ],
        meat=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.MEAT and x[i].x == 1
        ],
        vegetables=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.VEGETABLE and x[i].x == 1
        ],
        fruits=[
            ingredients[i].value
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.FRUIT and x[i].x == 1
        ],
    )


def maximize_taste_penalty_price(
    constraints_values: PizzaConstraintsValues,
    constraints_ingredients: PizzaConstraintsIngredients,
    lambda_param: float = 0.5,
) -> Pizza:
    # TODO: implement this function (description at the top of the file)
    # recomendation: use latex notation to describe the suggested model
    
    model = Model()

    # sets
    ingredients = [ingredient for ingredient in PizzaIngredients]
    ingredients_names = [ingredient.value.name for ingredient in ingredients]

    # TODO: rewrite to be cleaner
    # I made a workaround to define a taste variable direclty in each of the taste objects
    for ingredient in ingredients:
        ingredient_fat = ingredient.value.fat
        ingredient_type = ingredient.value.type.value

        if ingredient_type == IngredientType.DOUGH.value:
            ingredient.value.expected_taste =  0.05 * np.mean(ingredient_fat)
        if ingredient_type == IngredientType.SAUCE.value:
            ingredient.value.expected_taste =  0.2  * np.mean(ingredient_fat)
        if ingredient_type == IngredientType.CHEESE.value:
            ingredient.value.expected_taste =  0.3 * np.mean(ingredient_fat)
        if ingredient_type == IngredientType.FRUIT.value:
            ingredient.value.expected_taste =   0.1 * np.mean(ingredient_fat)
        if ingredient_type == IngredientType.MEAT.value:
            ingredient.value.expected_taste =  0.3 * np.mean(ingredient_fat)
        if ingredient_type == IngredientType.VEGETABLE.value:
            ingredient.value.expected_taste =  0.05 * np.mean(ingredient_fat)

    # variables
    x = [
        model.add_var(var_type=INTEGER, lb=0, ub=1, name=ingredient)
        for ingredient in ingredients_names
    ]

    # obj = Expected_value(taste(pizza)) - lambda * price(pizza), where lambda is a parameter that will be provided by the maestro pizza maker
    expected_value_taste_pizza = xsum(x[i] * ingredients[i].value.expected_taste for i in range(len(ingredients))) / len(ingredients)
    price_pizza = xsum(x[i] * ingredients[i].value.price for i in range(len(ingredients)))
    model.objective = maximize(
        expected_value_taste_pizza - lambda_param * price_pizza
    )

    # constraints
    model += (
        xsum(x[i] * ingredients[i].value.protein for i in range(len(ingredients)))
        >= constraints_values.protein.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.protein for i in range(len(ingredients)))
        <= constraints_values.protein.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.fat.mean() for i in range(len(ingredients)))
        >= constraints_values.fat.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.fat.mean() for i in range(len(ingredients)))
        <= constraints_values.fat.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.carbohydrates for i in range(len(ingredients)))
        >= constraints_values.carbohydrates.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.carbohydrates for i in range(len(ingredients)))
        <= constraints_values.carbohydrates.max
    )
    model += (
        xsum(x[i] * ingredients[i].value.calories for i in range(len(ingredients)))
        >= constraints_values.calories.min
    )
    model += (
        xsum(x[i] * ingredients[i].value.calories for i in range(len(ingredients)))
        <= constraints_values.calories.max
    )

    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.DOUGH
        )
        == constraints_ingredients.dough
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.SAUCE
        )
        == constraints_ingredients.sauce
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.CHEESE
        )
        == constraints_ingredients.cheese
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.MEAT
        )
        == constraints_ingredients.meat
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.VEGETABLE
        )
        == constraints_ingredients.vegetables
    )
    model += (
        xsum(
            x[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.FRUIT
        )
        == constraints_ingredients.fruits
    )

    # optimize
    model.optimize()

    # check solution
    if model.status != OptimizationStatus.OPTIMAL:
        raise Exception(
            "The model is not optimal -> likely no solution found (infeasible))"
        )

    # solution
    return Pizza(
        dough=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.DOUGH and x[i].x == 1
        ][0],
        sauce=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.SAUCE and x[i].x == 1
        ][0],
        cheese=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.CHEESE and x[i].x == 1
        ],
        meat=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.MEAT and x[i].x == 1
        ],
        vegetables=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.VEGETABLE and x[i].x == 1
        ],
        fruits=[
            ingredients[i]
            for i in range(len(ingredients))
            if ingredients[i].value.type == IngredientType.FRUIT and x[i].x == 1
        ],
    )
# class representing a pizza

from dataclasses import dataclass
from typing import List, Literal, Optional

from maestro_pizza_maker.ingredients import PizzaIngredients
from maestro_pizza_maker.ingredients import IngredientType

import numpy as np

import secrets
import random
from datetime import datetime

@dataclass
class Pizza:
    dough: Literal[
        PizzaIngredients.CLASSIC_DOUGH,
        PizzaIngredients.THIN_DOUGH,
        PizzaIngredients.WHOLEMEAL_DOUGH,
    ]
    sauce: Literal[PizzaIngredients.TOMATO_SAUCE, PizzaIngredients.CREAM_SAUCE]
    cheese: Optional[
        List[
            Literal[
                PizzaIngredients.MOZZARELA,
                PizzaIngredients.CHEDDAR,
                PizzaIngredients.PARMESAN,
            ]
        ]
    ] = None
    fruits: Optional[
        List[Literal[PizzaIngredients.PINEAPPLE, PizzaIngredients.APPLE]]
    ] = None
    meat: Optional[
        List[
            Literal[
                PizzaIngredients.HAM, PizzaIngredients.BACON, PizzaIngredients.SAUSAGE
            ]
        ]
    ] = None
    vegetables: Optional[
        List[
            Literal[
                PizzaIngredients.MUSHROOMS,
                PizzaIngredients.ONIONS,
                PizzaIngredients.PEPPER,
            ]
        ]
    ] = None

    def __post_init__(self) -> None:
        if self.cheese is None:
            self.cheese = []
        if self.fruits is None:
            self.fruits = []
        if self.meat is None:
            self.meat = []
        if self.vegetables is None:
            self.vegetables = []
        self.ingredients = [
            self.dough,
            self.sauce,
            *self.cheese,
            *self.fruits,
            *self.meat,
            *self.vegetables,
        ]

    @property
    def price(self) -> float:
        return sum(ingredient.value.price for ingredient in self.ingredients)

    @property
    def protein(self) -> float:
        return sum(ingredient.value.protein for ingredient in self.ingredients)

    @property
    def fat(self) -> np.array:
        return sum(ingredient.value.fat for ingredient in self.ingredients)

    @property
    def average_fat(self) -> float:
        # TODO: implement average fat calculation
        # HINT: check the `PizzaIngredients` class properly, you will find a `fat` property there which is a numpy array representing the drawings from the fat distribution
        # since fat is a random variable, we will calculate the average fat of the pizza by averaging the fat vectors of the ingredients

        # Solution Logic: Loop through each ingredient of the pizza, calculate the mean of its fat distribution, then take the mean of the means of the ingredients.

        # Comment: The pizza is the sum of it's ingredients, therefore the average fat of the pizza will be the average of the sum of its ingredients! 
        return np.mean([np.mean(ingredient.value.fat) for ingredient in self.ingredients])

    @property
    def carbohydrates(self) -> float:
        return sum(ingredient.value.carbohydrates for ingredient in self.ingredients)

    @property
    def calories(self) -> float:
        return sum(ingredient.value.calories for ingredient in self.ingredients)

    @property
    def name(self) -> str:
        # TODO: implement name generation, it is purely up to you how you want to do it
        # (you can use random, you can use some kind of algorithm) - just make sure that
        # the name is unique.

        # Solution Logic: Create a descriptive name for the pizza and smack a unique 
        # identifier at the end, also make sure that the user can't override the random seed
        #  by setting it equal to the current timestamp

        random.seed(datetime.now().timestamp())

        ingredient_list = []

        for ingredient in self.ingredients:
            ingredient_name = ingredient.value.name
            ingredient_type = ingredient.value.type.value

            if ingredient_type == IngredientType.DOUGH.value:
                dough_name = ingredient_name
            if ingredient_type  == IngredientType.SAUCE.value:
                sauce_name = ingredient_name
            if ingredient_type  != IngredientType.SAUCE.value and ingredient_type != IngredientType.DOUGH.value: 
                ingredient_list.append(ingredient_name)

        if len(ingredient_list) == 0:
            return f"{dough_name} PIZZA, WITH {sauce_name}, UNIQUE ORDER KEY: {str(secrets.token_hex(8))}"
        elif len(ingredient_list) == 1:
            return f"{dough_name} PIZZA, WITH {ingredient_list[0]} AND {sauce_name}, UNIQUE ORDER KEY: {str(secrets.token_hex(8))}"
        else:
            return f"{dough_name} PIZZA, WITH {', '.join(ingredient_list)} AND {sauce_name}, UNIQUE ORDER KEY: {str(secrets.token_hex(8))}"
        
    @property
    def taste(self) -> np.array:
        # TODO: implement taste function
        # The famous fact that taste is subjective is not true in this case. We believe that fat is the most important factor, since fat carries the most flavor.
        # So we will use the fat vector to calculate the taste of the pizza with the following formula:
        # taste = 0.05 * fat_dough + 0.2 * fat_sauce + 0.3 * fat_cheese + 0.1 * fat_fruits + 0.3 * fat_meat + 0.05 * fat_vegetables
        
        for index, ingredient in enumerate(self.ingredients):
            if index == 0:
                taste = np.zeros(np.shape(ingredient.value.fat))

            ingredient_fat = ingredient.value.fat
            ingredient_type = ingredient.value.type.value

            if ingredient_type == IngredientType.DOUGH.value:
                taste = np.add(taste, 0.05 * ingredient_fat)
            if ingredient_type == IngredientType.SAUCE.value:
                taste =  np.add(taste, 0.2  * ingredient_fat)
            if ingredient_type == IngredientType.CHEESE.value:
                taste =  np.add(taste, 0.3 * ingredient_fat)
            if ingredient_type == IngredientType.FRUIT.value:
                taste =  np.add(taste, 0.1 * ingredient_fat)
            if ingredient_type == IngredientType.MEAT.value:
                taste =  np.add(taste, 0.3 * ingredient_fat)
            if ingredient_type == IngredientType.VEGETABLE.value:
                taste =  np.add(taste, 0.05 * ingredient_fat)
        
        return taste



# class representing the pizza menu

from dataclasses import dataclass
from typing import List

import pandas as pd

from maestro_pizza_maker.pizza import Pizza


@dataclass
class PizzaMenu:
    pizzas: List[Pizza]

    def to_dataframe(self, sort_by: str, descendent: bool) -> pd.DataFrame:
        # TODO: transform the list of pizzas into a pandas dataframe, where each row represents a pizza
        # and it contains the following columns: name, price, protein, average_fat, carbohydrates, calories and ingredients
        # where ingredients is a list of ingredients.
        # The dataframe should be sorted by the column specified by the sort_by parameter
        # and the order of sorting should be specified by the descendent parameter
        # (descendent=True means that the dataframe should be sorted in a descendent order)
        #
        # Example:
        #
        # pizza_menu = PizzaMenu(pizzas=[Pizza(sauce=PizzaIngredients.CREAM_SAUCE, dough=PizzaIngredients.CLASSIC_DOUGH)])
        # pizza_menu.to_dataframe(sort_by="price", descendent=True)
        #
        # should return a dataframe with a single row and the following columns:
        # name, price, protein, average_fat, carbohydrates, calories, ingredients
        # where the name column contains the name of the pizza, price contains the price of the pizza,
        # protein contains the protein content of the pizza, average_fat contains the average_fat content of the pizza,
        # carbohydrates contains the carbohydrates content of the pizza, calories contains the calories content of the pizza
        # and ingredients contains a list of ingredients that the pizza contains
        #
        # The dataframe should be sorted by the price column in a descendent order

        df_pizza_menu = pd.DataFrame(columns = ["name", "price", "protein", "average_fat", "carbohydrates", "calories", "ingredients"])
        for pizza in self.pizzas:
            ingredient_list = [ingredient.value.name for ingredient in pizza.ingredients]
            df_new_row = pd.DataFrame({"name": pizza.name, "price": pizza.price, "protein": pizza.protein, "average_fat": pizza.average_fat,
                "carbohydrates": pizza.carbohydrates, "calories": pizza.calories, "ingredients": [ingredient_list]})
            df_pizza_menu = pd.concat([df_pizza_menu, df_new_row], ignore_index=True) 
        df_pizza_menu.sort_values(by=sort_by, ascending= not descendent, inplace=True)
        return df_pizza_menu

    @property
    def cheapest_pizza(self) -> Pizza:
        # TODO: return the cheapest pizza from the menu
        return min(self.pizzas, key=lambda x: x.price)

    @property
    def most_expensive_pizza(self) -> Pizza:
        # TODO: return the cheapest pizza from the menu
        return max(self.pizzas, key=lambda x: x.price)

    @property
    def least_caloric_pizza(self) -> Pizza:
        # TODO: return the most caloric pizza from the menu
        return min(self.pizzas, key=lambda x: x.calories)

    @property
    def most_caloric_pizza(self) -> Pizza:
        # TODO: return the most caloric pizza from the menu
        return max(self.pizzas, key=lambda x: x.calories)

    @property
    def least_tasty_pizza(self) -> Pizza:
        # TODO: return the most caloric pizza from the menu
        return min(self.pizzas, key=lambda x: x.taste)

    @property
    def most_tasty_pizza(self) -> Pizza:
        # TODO: return the most caloric pizza from the menu
        return max(self.pizzas, key=lambda x: x.taste)

    def get_most_fat_pizza(self, quantile: float = 0.5) -> Pizza:
        # TODO: return the most fat pizza from the menu
        # consider the fact that fat is random and it is not always the same, so you should return the pizza that has the most fat in the quantile of cases specified by the quantile parameter
        return max(self.pizzas, key=lambda x: np.quantile(x.fat, quantile))

    def add_pizza(self, pizza: Pizza) -> None:
        # TODO: code a function that adds a pizza to the menu
        self.pizzas.append(pizza)

    def remove_pizza(self, pizza: Pizza) -> None:
        # TODO: code a function that removes a pizza from the menu
        # do not forget to check if the pizza is actually in the menu
        # if it is not in the menu, raise a ValueError
        if pizza in self.pizzas:
            self.pizzas.remove(pizza)
        else:
            raise ValueError("Could not find pizza in the menu!")

    def __len__(self) -> int:
        # TODO: return the number of pizzas in the menu
        return len(self.pizzas)

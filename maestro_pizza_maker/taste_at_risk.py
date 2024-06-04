# The maestro pizza maker is aware of the fact that the fat content of the ingredients is random and it is not always the same.
# Since fat is the most important factor in taste, the maestro pizza maker wants to know how risky his pizza menu is.

# TODO: define 2 risk measures for the pizza menu and implement them (1 - Taste at Risk (TaR), 2 - Conditional Taste at Risk (CTaR), also known as Expected Shorttaste (ES)

from maestro_pizza_maker.pizza import Pizza
from maestro_pizza_maker.pizza_menu import PizzaMenu

import numpy as np

def taste_at_risk_pizza(pizza: Pizza, quantile: float) -> float:
    # TODO: implement the taste at risk measure for a pizza
    # quantile is the quantile that we want to consider
    # Hint: Similarity between the Taste at Risk and the Value at Risk is not a coincidence or is it?
    # Hint: Use function taste from Pizza class, but be aware that the higher the taste, the better -> the lower the taste, the worse
    
    # Solution Logic: 
    # General VaR calculation would be to calculate the distribution of daily returns, with the aim of understanding the maximum potential daily LOSS of a portfolio
    # at a given confidence level. Therefore, we either "flip the sign of the distribution" to get the loss distribution (losses on right hand side of distribution instead 
    # of left), the user then specifices a x quantile (x*100% percentile) to consider and we ouput x*100% of losses are below the x*100% VaR.
    # In our TaR case this is different, my interpetation would be that the user wants to know, at a given quantile (i.e. 5th percentile), how many taste values fall 
    # below that value. Therefore, the function will output the taste value at which i.e. 5% of the taste values are less than that TaR, therefore there is no need to "flip the sign".

    return np.quantile(pizza.taste, quantile)

def taste_at_risk_menu(menu: PizzaMenu, quantile: float) -> float:
    # TODO: implement the taste at risk measure for a menu
    # quantile is the quantile that we want to consider
    # Hint: the taste of the whole menu is the sum of the taste of all pizzas in the menu, or? ;)

    # Solution Logic: 
    # In the VaR case, we can sum the returns of the individual elements of the portfolio to get the return of the portfolio,
    # and since we have a very similar case for TaR, I would say that we could then simply sum the taste of the pizzas and 
    # reap the benefits of a well diversified pizza menu ;)

    sum_tastes_menu = np.sum([pizza.taste for pizza in menu.pizzas], axis=0)
    return np.quantile(sum_tastes_menu, quantile)

def conditional_taste_at_risk_pizza(pizza: Pizza, quantile: float) -> float:
    # TODO: implement the conditional taste at risk measure for a pizza
    # quantile is the quantile that we want to consider
    # Hint: Simmilarity between the Conditional Taste at Risk and the Conditional Value at Risk is not a coincidence or is it?
    
    # Solution Logic: 
    # In the conditional VaR case, our aim is to calculate the average of the losses that exceed the VaR quantile.
    # In the conditional TaR case, my assumption is that we need to first calculate the TaR at a given quantile, then we calculate the average of the taste
    # that is less than that quantile.

    TaR_pizza = taste_at_risk_pizza(pizza, quantile)
    tastes_below_TaR = pizza.taste[pizza.taste <= TaR_pizza]
    return tastes_below_TaR.mean()

def conditional_taste_at_risk_menu(menu: PizzaMenu, quantile: float) -> float:
    # TODO: implement the conditional taste at risk measure for a menu
    # Hint: the taste of the whole menu is the sum of the taste of all pizzas in the menu, or? ;) (same as for the taste at risk)

    # Solution Logic: 
    # Very similar to the cTaR solution of the individual pizza, but we sum the taste of the pizzas to get the taste of the menu
    
    sum_tastes_menu = np.sum([pizza.taste for pizza in menu.pizzas], axis=0)
    TaR_menu = taste_at_risk_menu(menu, quantile)
    tastes_below_TaR = sum_tastes_menu[sum_tastes_menu <= TaR_menu]
    return tastes_below_TaR.mean()

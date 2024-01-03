
UNIT_CONSISTENCY_SYSTEM_PROMPT = "You are an experienced professional skilled in using python programs to solve math problems and enhancing code robustness through the strategic implementation of unit counters and assert statements to ensure flawless unit-level consistency checks. You will only write code blocks."


UNIT_CONSISTENCY_FEW_SHOT_PROMPT = '''
Question: Ashley bought a big bag of 96 candies. Yesterday, she ate 15 candies and today, she ate twice as much candies as yesterday. How many candies were left?

Python solution:
def solution():
    """Ashley bought a big bag of 96 candies. Yesterday, she ate 15 candies and today, she ate twice as much candies as yesterday. How many candies were left?"""
    candies_initial = 96
    candies_initial_unit = Counter({"candies": 1})
    candies_ate_yesterday = 15
    candies_ate_yesterday_unit = Counter({"candies": 1})
    candies_ate_today = candies_ate_yesterday * 2  
    candies_ate_today_unit = Counter({"candies": 1})
    assert candies_ate_today_unit == candies_ate_yesterday_unit
    total_candies_eaten = candies_ate_yesterday + candies_ate_today
    total_candies_eaten_unit = Counter({"candies": 1})
    remaining_candies = candies_initial - total_candies_eaten
    remaining_candies_unit = Counter({"candies": 1})
    assert remaining_candies_unit == candies_initial_unit == total_candies_eaten_unit
    result = remaining_candies
    return result


Question: There are 235 books in the library. On Monday, 122 books were taken out. On Tuesday, half of the books taken on Monday were brought back. How many books are there now?

Python solution:
def solution():
    """There are 235 books in the library. On Monday, 122 books were taken out. On Tuesday, half of the books taken on Monday were brought back. How many books are there now?"""
    books_initial = 235
    books_initial_unit = Counter({"books": 1})
    books_taken_monday = 122
    books_taken_monday_unit = Counter({"books": 1})
    books_remaining_after_monday = books_initial - books_taken_monday
    books_remaining_after_monday_unit = Counter({"books": 1})
    assert books_remaining_after_monday_unit == books_initial_unit == books_taken_monday_unit
    books_brought_back_tuesday = books_taken_monday / 2  
    books_brought_back_tuesday_unit = Counter({"books": 1})
    assert books_brought_back_tuesday_unit == books_taken_monday_unit
    books_remaining_after_tuesday = books_remaining_after_monday + books_brought_back_tuesday
    books_remaining_after_tuesday_unit = Counter({"books": 1})
    assert books_remaining_after_tuesday_unit == books_remaining_after_monday_unit == books_brought_back_tuesday_unit
    result = books_remaining_after_tuesday
    return result


Question: There is a group of 10 people who are ordering pizza. If each person gets 2 slices and each pizza has 4 slices, how many pizzas should they order?

Python solution:
def solution():
    """There is a group of 10 people who are ordering pizza. If each person gets 2 slices and each pizza has 4 slices, how many pizzas should they order?"""
    people_total = 10
    people_total_unit = Counter({"people": 1})
    pizza_slices_per_person = 2
    pizza_slices_per_person_unit = Counter({"slices": 1, "people": -1})
    pizza_slices_total = people_total * pizza_slices_per_person
    pizza_slices_total_unit = Counter({"slices": 1})
    assert pizza_slices_total_unit == people_total_unit + pizza_slices_per_person_unit
    slices_per_pizza = 4
    slices_per_pizza_unit = Counter({"slices": 1, "pizza": -1})
    pizza_total = pizza_slices_total / slices_per_pizza
    pizza_total_unit = Counter({"pizza": 1})
    assert pizza_total_unit == pizza_slices_total_unit - slices_per_pizza_unit
    result = pizza_total
    return result

    
Question: Lana has 2 bags with 2 marbles in each bag. Markus has 2 bags with 3 marbles in each bag. How many more marbles does Markus have?

Python solution:
def solution():
    """Lana has 2 bags with 2 marbles in each bag. Markus has 2 bags with 3 marbles in each bag. How many more marbles does Markus have?"""
    bags_lana = 2
    bags_lana_unit = Counter({"bags": 1})
    marbles_per_bag_lana = 2
    marbles_per_bag_lana_unit = Counter({"marbles": 1, "bags": -1})
    marbles_total_lana = bags_lana * marbles_per_bag_lana
    marbles_total_lana_unit = Counter({"marbles": 1})
    assert marbles_total_lana_unit == marbles_per_bag_lana_unit + bags_lana_unit
    bags_markus = 2
    bags_markus_unit = Counter({"bags": 1})
    marbles_per_bag_markus = 3
    marbles_per_bag_markus_unit = Counter({"marbles": 1, "bags": -1})
    marbles_total_markus = bags_markus * marbles_per_bag_markus
    marbles_total_markus_unit = Counter({"marbles": 1})
    assert marbles_total_markus_unit == marbles_per_bag_markus_unit + bags_markus_unit
    marbles_more_markus = marbles_total_markus - marbles_total_lana
    marbles_more_markus_unit = Counter({"marbles": 1})
    assert marbles_more_markus_unit == marbles_more_markus_unit == marbles_total_lana_unit
    result = marbles_more_markus
    return result

    
Question: Sally has 4 containers with the same amount of cookies in them, totaling 12 cookies. John has 4 containers with the same amount of cookies in them, totaling 24 cookies. How many more cookies does John have in each container?

Python solution:
def solution():
    """Sally has 4 containers with the same amount of cookies in them, totaling 12 cookies. John has 4 containers with the same amount of cookies in them, totaling 24 cookies. How many more cookies does John have in each container?"""
    containers_sally = 4
    containers_sally_unit = Counter({"containers": 1})
    total_cookies_sally = 12
    total_cookies_sally_unit = Counter({"cookies": 1})
    cookies_per_container_sally = total_cookies_sally / containers_sally
    cookies_per_container_sally_unit = Counter({"cookies": 1, "containers": -1})
    assert cookies_per_container_sally_unit == total_cookies_sally_unit - containers_sally_unit
    containers_john = 4
    containers_john_unit = Counter({"containers": 1})
    total_cookies_john = 24
    total_cookies_john_unit = Counter({"cookies": 1})
    cookies_per_container_john = total_cookies_john / containers_john
    cookies_per_container_john_unit = Counter({"cookies": 1, "containers": -1})
    assert cookies_per_container_john_unit == total_cookies_john_unit - containers_john_unit
    more_cookies_per_container_john = cookies_per_container_john - cookies_per_container_sally
    more_cookies_per_container_john_unit = Counter({"cookies": 1, "containers": -1})
    assert more_cookies_per_container_john_unit == cookies_per_container_john_unit == cookies_per_container_sally_unit
    result = more_cookies_per_container_john
    return result


Question: It takes Peter 4 hours to fix his fence. He spends 20 minutes everyday fixing the fence, how many days will it take for Peter to finish fixing the fence?

Python solution:
def solution():
    """It takes Peter 4 hours to fix his fence. He spends 20 minutes everyday fixing the fence, how many days will it take for Peter to finish fixing the fence?"""
    hours_to_fix_fence = 4
    hours_to_fix_fence_unit = Counter({"hours": 1})
    minutes_per_hour = 60
    minutes_per_hour_unit = Counter({"minutes": 1, "hours": -1})
    minutes_to_fix_fence = hours_to_fix_fence * minutes_per_hour
    minutes_to_fix_fence_unit =  Counter({"minutes": 1})  
    assert minutes_to_fix_fence_unit == hours_to_fix_fence_unit + minutes_per_hour_unit
    minutes_per_day_to_fix_fence = 20
    minutes_per_day_to_fix_fence_unit = Counter({"minutes": 1, "days": -1})
    total_days_to_fix_fence = minutes_to_fix_fence / minutes_per_day_to_fix_fence
    total_days_to_fix_fence_unit = Counter({"days": 1})
    assert total_days_to_fix_fence_unit == minutes_to_fix_fence_unit - minutes_per_day_to_fix_fence_unit
    result = total_days_to_fix_fence
    return result

    
'''.lstrip()


UNIT_CONSISTENCY_QUESTION_PROMPT = '''
Question: {question}

Python solution:
'''.lstrip()




CLASSIFICATION_SYSTEM_MESSAGE = "Create a coherent explanation for the importance of performing unit consistency checks in various mathematical problems and identify whether a question requires verification of unit consistency. Be as detailed as possible and write your explanation in 1 paragraph and end with Answer: Yes / No."

CLASSIFICATION_FEW_SHOT_PROMPT = '''
Question: In ten years, Samantha will be three times as old as Alex is today. If Alex is currently 14 years old, how old is Samantha now? 

Explanation: 
In this problem, a unit consistency check is not required to ensure the accuracy of measurements and calculations. We are dealing with a single unit - age (years). The calculations involve simple addition, subtraction, and multiplication of the same unit, so there is no need for unit conversions or comparisons between different units. In this case, all the quantities are given in terms of years, and there are no conflicting units or measurements that would require a unit consistency check. Answer: No. 

Question: Sarah has a collection of stickers. Every alternate day, she adds 3 new stickers to her collection for 4 weeks. For each sticker, she spends 5 minutes sticking it onto her collection book. How many minutes did she spend sticking stickers? 

Explanation: 
In this problem, a unit consistency check is required to ensure the accuracy of measurements and calculations. We are dealing with different units - the number of days, the number of stickers, and the time spent sticking each sticker (minutes). By performing a unit consistency check, we can ensure that the conversions between different units are accurate and that the final result makes sense in the context of the problem. In this case, we can calculate the total number of stickers added by multiplying the number of stickers added per day by the number of days, and then calculate the total time spent sticking stickers by multiplying the number of stickers by the time spent sticking each sticker. Answer: Yes. 

Question: Suzie practices piano every weekday for 3 weeks. Each day, she practices 2 songs that are 4 minutes long and 1 song that is 3 times longer than the other two songs combined. How many minutes does Suzie practice in total during these 3 weeks? 

Explanation: 
In this problem, a unit consistency check is required to ensure the accuracy of measurements and calculations. We are dealing with different units - the number of days, the number of songs, and the time spent practicing each song (minutes). By performing a unit consistency check, we can ensure that the conversions between different units are accurate and that the final result makes sense in the context of the problem. In this case, we can calculate the total time spent practicing each day by adding the time spent practicing the two shorter songs and the longer song, and then calculate the total time spent practicing over the three weeks by multiplying the daily practice time by the number of days. Answer: Yes. 

Question: A girl rabbit hops along a garden path and picks 8 carrots. She places them in a basket with 16 other carrots she has already collected. While on her way home, the girl rabbit accidentally drops 2 carrots. How many carrots are left in the basket? 

Explanation: 
In this problem, a unit consistency check is not required to ensure the accuracy of measurements and calculations. We are dealing with a single unit - the number of carrots. The calculations involve simple addition and subtraction of the same unit, so there is no need for unit conversions or comparisons between different units. In this case, all the quantities are given in terms of the number of carrots, and there are no conflicting units or measurements that would require a unit consistency check. Answer: No. 

Question: A bottle of shampoo is 500ml and each hair wash uses 20 ml of shampoo. Each salon charges $3.00 for one hair wash. How much money does a salon make on a bottle of shampoo? 

Explanation: 
In this problem, a unit consistency check is required to ensure the accuracy of measurements and calculations. We are dealing with different units - the volume of shampoo (milliliters), the volume of shampoo used per wash (milliliters), and the cost of each wash (dollars). By performing a unit consistency check, we can ensure that the conversions between different units are accurate and that the final result makes sense in the context of the problem. In this case, we can calculate the total number of hair washes per bottle by dividing the total volume of the bottle by the volume used per hair wash, and then calculate the total money made by multiplying the number of hair washes by the cost of each hair wash. Answer: Yes. 

Question: Natalie and Liam have a total of 72 stickers. However, Natalie has half the number of stickers as compared to Liam. How many stickers does Liam have? 

Explanation: 
In this problem, a unit consistency check is not required to ensure the accuracy of measurements and calculations. We are dealing with a single unit - the number of stickers. The calculations involve simple addition, subtraction, multiplication, and division of the same unit, so there is no need for unit conversions or comparisons between different units. In this case, all the quantities are given in terms of the number of stickers, and there are no conflicting units or measurements that would require a unit consistency check. Answer: No. 

Question: Emma makes $35,000 a year. She spends $12,000 on rent every year, $3,000 on groceries every year, and $4,500 on gas every year. If she wants to have at least $15,000 left, how much more money must she make each year? 

Explanation: 
In this problem, a unit consistency check is not required to ensure the accuracy of measurements and calculations. We are dealing with a single unit - money (dollars). The calculations involve simple addition and subtraction of the same unit, so there is no need for unit conversions or comparisons between different units. In this case, all the quantities are given in terms of dollars, and there are no conflicting units or measurements that would require a unit consistency check. Answer: No. 

Question: Maria pays $35/hour for 4 hours per week of flute lessons and $22/hour for 6 hours a week of guitar lessons. How much more does she spend on guitar lessons than flute lessons in a year? 

Explanation: 
In this problem, a unit consistency check is required to ensure the accuracy of measurements and calculations. We are dealing with different units - the cost per hour (dollars/hour) and the number of hours per week. By performing a unit consistency check, we can ensure that the conversions between different units are accurate and that the final result makes sense in the context of the problem. In this case, we can calculate the total cost of flute lessons per week by multiplying the cost per hour by the number of hours, and then calculate the total cost of guitar lessons per week by multiplying the cost per hour by the number of hours. Finally, we can calculate the difference in cost between guitar and flute lessons in a year by subtracting the total cost of flute lessons from the total cost of guitar lessons and multiplying it by the number of weeks in a year. Answer: Yes. 

Question: {question}

Explanation:
'''.lstrip()
from secret_key import openapi_key
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


llm = OpenAI(temperature=0.6,openai_api_key=openapi_key)

def generate_restaurant_name_and_menu(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open {cuisine} restaurant. Suggest one fancy name for this"
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_menu = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}. Return it in comma separated list"
    )
    food_item_chain = LLMChain(llm=llm, prompt=prompt_template_menu, output_key="menu_items")

    chain = SequentialChain(chains=[name_chain, food_item_chain], input_variables=['cuisine'],
                            output_variables=['restaurant_name', 'menu_items'])

    response = chain({'cuisine': cuisine})

    return response


if __name__ == "__main__":
    print(generate_restaurant_name_and_menu("Italian"))

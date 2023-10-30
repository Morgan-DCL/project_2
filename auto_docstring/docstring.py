
import openai
import inspect
import os
import json


ai_description = (
    "You are an AI expert in Python documentation writing, " +
    "familiar with PEP8 standards," +
    "including the rule of keeping lines under 79 characters." +
    "Black Formated Python Style." +
    "You must understand the function and draft its documentation " +
    "in numpy-style docstrings." +
    "Do not copy the actual code again only the docstring." +
    "No need for additional notes and examples. " +
    "The main description should be placed" +
    "at the beginning without using a 'description' title."
)


class DocString():
    """
    A class used to generate documentation for a given function
    using OpenAI's GPT-4 model.

    Attributes
    ----------
    fonction : function
        The function for which the documentation is to be
        generated.
    ai_description : str
        The description of the AI model used for generating the
        documentation.
    model : str, optional
        The model used for generating the documentation
        (default is 'gpt-4').
    temperature : float, optional
        The temperature parameter for the model
        (default is 0).
    max_tokens : int, optional
        The maximum number of tokens for the model to
        generate (default is 1500).
    langue : str, optional
        The language in which the documentation is to be
        generated (default is 'French').

    Methods
    -------
    _generate_prompt():
        Generates the prompt for the OpenAI model.
    ask_openai():
        Sends the generated prompt to the OpenAI model and returns
        the generated documentation.
    print_in(msg):
        Writes the generated documentation into a file.
    """

    def __init__(
        self, fonction,
        ai=ai_description,
        model="gpt-4",
        temperature=0,
        max_tokens=1500,
        langue: str = "French"
    ):
        """
        Constructs all the necessary attributes for the
        DocString object.

        Parameters
        ----------
        fonction : function
            The function for which the documentation
            is to be generated.
        ai_description : str
            The description of the AI model used for
            generating the documentation.
        model : str, optional
            The model used for generating the
            documentation (default is 'gpt-4').
        temperature : float, optional
            The temperature parameter for the
            model (default is 0).
        max_tokens : int, optional
            The maximum number of tokens for the
            model to generate (default is 1500).
        langue : str, optional
            The language in which the documentation
            is to be generated (default is 'French').
        """

        self.fonction = fonction
        self.ai_description = ai
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.langue = langue
        openai.api_key = "" ##################################################################

    def _generate_prompt(self):
        """
        Generates the prompt for the OpenAI model.

        Returns
        -------
        str
            The generated prompt.
        """

        if inspect.isclass(self.fonction):
            function_str = inspect.getsource(self.fonction)
        elif inspect.isfunction(self.fonction):
            function_str = inspect.getsource(self.fonction)
        else:
            raise ValueError("L'objet donné n'est ni une classe ni une fonction")
        cmt = "Document this function carefully in "
        return (
            f"{cmt} {self.langue}: \n{function_str}\n " +
            "without a return example."
        )

    def ask_openai(self):
        """
        Sends the generated prompt to the OpenAI model and
        returns the generated documentation.

        Returns
        -------
        str
            The generated documentation.
        """

        print("Generating Docsting....")
        prompt_str = self._generate_prompt()

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": ai_description,
                    },
                    {
                        "role": "user",
                        "content": prompt_str
                    }
                ]
            )
            msg = response['choices'][0]['message']['content']
            self.print_in(msg)
            return msg

        except Exception as e:
            print(f"Error: {e}")
            return None

    def print_in(self, msg):
        """
        Writes the generated documentation into a file.

        Parameters
        ----------
        msg : str
            The generated documentation.
        """
        fld_nm : str = '_docstring'
        output_filename = (
            f"docstring_{self.langue}_{self.fonction.__name__}.txt"
        )
        # Écrire le message dans le fichier
        with open(output_filename, "w", encoding="utf-8") as file:
            file.write(msg)

# ? Exemple d'utilisation :
# ? doc = DocString(test_function)
# ? print(doc.ask_openai())

# def calc_salary(salaire):
#     if salaire <= 0:
#         raise ValueError("Ca pu la merde")
#     return salaire * (1 + 0.25)

# doc = DocString(DocString, langue="English")
# print(doc.ask_openai())
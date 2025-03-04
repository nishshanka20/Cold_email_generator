import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm=ChatGroq(temperature=0,groq_api_key=os.getenv("GROQ_API_KEY"),model_name="llama-3.3-70b-versatile")

    def extract_jobs(self,cleaned_text):
        prompot_extract = PromptTemplate.from_template(
            """
            ### SCARED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's of a website;
            tour job is to extract the job posting and return them in JSON fromat containing the
            following keys: `role`,`experience`,`skils` and `description`.
            only return the valid JSON.
            ### VALID JSON(NO PREAMBLE):
            """
        )

        chain_extract = prompot_extract | self.llm
        res = chain_extract.invoke(input={'page_data': cleaned_text})
        print("res content typ",type(res.content))
        try:
            json_parser=JsonOutputParser()
            res=json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big.unable to parse jobs.")
        return res if isinstance(res,list) else[res]

    def write_mail(self,job,links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Nishshanka, a developer in IT faculty university of Moratuwa. IT faculty is a big tech faculty  dedicated to facilitating
            the seamless integration of business processes via knowlegable studens. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of faculty 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase faculty portfolio: {link_list}
            Remember you are Nishshanka, developer at faculty. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ =="__main__":
    print(os.getenv("GROQ_API_KEY"))
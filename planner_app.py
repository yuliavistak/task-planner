# streamlit run planner_app.py

import streamlit as st
import json
import datetime
from datetime import datetime
import re
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import openai
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from scipy.spatial.distance import cosine
import copy

st.set_page_config(page_title="🗂️ Task Planner", layout="wide")

openai_api_key = 'INSERT_KEY_HERE'
openai.api_key = openai_api_key

if not openai_api_key:
    st.sidebar.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

def display_schedules_table_aggrid(employee_data, schedules):
    """
    Displays a schedules table using AgGrid with the following columns:
    Id, People, Start of Working Day, Busy, End of Working Day
    """
    
    df_people = pd.DataFrame(employee_data)
    df_schedules = pd.DataFrame(schedules)
    
    df_merged = pd.merge(df_people, df_schedules, on='id', how='left')
    
    df_merged.fillna({'name': 'Unknown', 'surname': 'Unknown'}, inplace=True)
    
    df_merged['People'] = df_merged['name'] + ' ' + df_merged['surname']
    
    df_merged['Start of Working Day'] = df_merged['start_at'].apply(lambda x: x if pd.notnull(x) else "N/A")
    df_merged['End of Working Day'] = df_merged['end_at'].apply(lambda x: x if pd.notnull(x) else "N/A")
    
    def format_busy(busy_list):
        if isinstance(busy_list, list) and busy_list:
            return ', '.join([f"{start} - {end}" for start, end in busy_list])
        else:
            return "None"
    
    df_merged['Busy'] = df_merged['busy'].apply(format_busy)
    
    df_final = df_merged[['id', 'People', 'Start of Working Day', 'Busy', 'End of Working Day']].rename(columns={
        'id': 'ID',
        'People': 'People',
        'Start of Working Day': 'Start of Working Day',
        'Busy': 'Busy',
        'End of Working Day': 'End of Working Day'
    })
    
    gb = GridOptionsBuilder.from_dataframe(df_final)
    gb.configure_pagination(paginationAutoPageSize=True)  
    gb.configure_side_bar() 
    
    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        wrapText=True,
        autoHeight=True
    )
    
    gb.configure_column("ID", header_name="ID", width=70, maxWidth=100, pinned='left')
    gb.configure_column("People", header_name="People", width=200, maxWidth=250)
    gb.configure_column("Start of Working Day", header_name="Start of Working Day", width=150, maxWidth=200)
    gb.configure_column("Busy", header_name="Busy", width=300, maxWidth=350)
    gb.configure_column("End of Working Day", header_name="End of Working Day", width=150, maxWidth=200)
    
    gridOptions = gb.build()
    
    st.subheader("Updated Schedules")
    AgGrid(
        df_final,
        gridOptions=gridOptions,
        enable_enterprise_modules=False,
        height=400,
        fit_columns_on_grid_load=True,
        theme='streamlit', 
    )

@st.cache_data
def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from file: {file_path}")
        st.stop()

employee_data = load_json("peopleskills.json")
schedules = load_json("idschedule.json")
reusable = load_json("reusable.json")
disposable = load_json("disposable.json")

def get_unique_skills(employee_data):
    skills = set()
    for p in employee_data:
        skill = p['skills']
        for s in skill:
            skills.add(s)

    return list(skills)

skills = get_unique_skills(employee_data)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
docs_after_split_skills = [Document(page_content=chunk, metadata={"category": "Skills"}) for chunk in skills]
db_skills = FAISS.from_documents(
    docs_after_split_skills,
    embedding_model
)

def get_resources(reusable, disposable):
    resources = list(reusable.keys()) + list(disposable.keys())
    return resources

resources = get_resources(reusable, disposable)

docs_resources = [Document(page_content=chunk, metadata={"category": "Resources"}) for chunk in resources]
db_resources = FAISS.from_documents(docs_resources, embedding_model)
from scipy.spatial.distance import cosine

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def find_resource(name, bounds, data):

    choice = ''
    task_start = datetime.strptime(bounds[0], "%H:%M").time()
    task_end = datetime.strptime(bounds[1], "%H:%M").time()

    for resource_num in data[name]:

      availability = False
      if not data[name][resource_num]:
        availability = True

      for busy_time in data[name][resource_num]:
        busy_start = datetime.strptime(busy_time[0], "%H:%M").time()
        busy_end = datetime.strptime(busy_time[1], "%H:%M").time()
        if ((busy_end <= task_start) & (busy_end < task_end)) | ((busy_start > task_start) & (busy_start >= task_end)):
          availability = True
        else:
          availability = False
          break
      if availability == True:
        choice = resource_num
        break

    if choice != '':
      data[name][choice].append(tuple([bounds[0], bounds[1]]))
      return data
    else:
      return False
    
def get_relevant_resources(tasks):
  global reusable
  for p in tasks:
    if p['material_resources'] == []:
      p['real_resources'] = []
      continue

    real_resources = []
    for resource in p['material_resources']:
      rel_resource = resource['resource']
      rel_quantity = resource['quantity']

      input_embedding = embedding_model.embed_query(rel_resource)
      result = db_resources.similarity_search_by_vector(input_embedding, k=1)

      resource_name = result[0].page_content
      resource_embedding = embedding_model.embed_query(resource_name)
      similarity = cosine_similarity(input_embedding, resource_embedding)

      if similarity >= 0.9:

        if resource_name in disposable:
          if rel_quantity <= disposable[resource_name]:
            keys = {'resource': resource_name, 'quantity': rel_quantity, 'lack': 0, 'similarity': similarity, 'embedding': resource_name}
            real_resources.append(keys)
            disposable[resource_name] -= rel_quantity
          else:
            lack = rel_quantity - disposable[resource_name]
            keys = {'resource': None, 'quantity': rel_quantity, 'lack': lack, 'similarity': similarity, 'embedding': resource_name}
            real_resources.append(keys)

        # Reusable
        else:
          reusable_new = copy.deepcopy(reusable)
          availability = 0

          for i in range(rel_quantity):
            result = find_resource(resource_name, (p['start_time'], p['end_time']), reusable_new)
            if not isinstance(result, bool):
              availability += 1
              reusable_new = result

          if availability == rel_quantity:
            reusable = reusable_new
            keys = {'resource': resource_name, 'quantity': rel_quantity, 'lack': 0, 'similarity': similarity, 'embedding': resource_name}
            real_resources.append(keys)
          else:
            lack = rel_quantity - availability
            keys = {'resource': None, 'quantity': rel_quantity, 'lack': lack, 'similarity': similarity, 'embedding': resource_name}
            real_resources.append(keys)

      else:
        keys = {'resource': None, 'quantity': rel_quantity, 'lack': 0, 'similarity': similarity, 'embedding': resource_name}
        real_resources.append(keys)

      p['real_resources'] = real_resources

def structure_final_output(content, history=None):
    prompt = f"""
You are a helpful assistant that helps in structuring the text.

Text:{content}

Please, structure the text you got into the following format:

'Task 1: [Brief explanation of the task] - [Time bounds]
 The needed number of people: [number_of_people]
 Relevant Skills: [Skill 1, Skill 2, Skill 3]
 Material Resources: [Resource 1 (Quantity), Resource 2 (Quantity), Resource 3 (Quantity)]'

Notes about the format:
- in the second line the number of people should only be a number, not other text is allowed.
- if in line, where resources are defined, there is no format Resource 1 (Quantity), then simply write Material Resources: ''.
  Also for material resources there must be defined each resource separately('Notepad and pen (1 each)' is a bad case, must be 'Notepad (1), Pen (1)')

I want to see only structured text without any of your comments(do not include quotation marks).
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that helps in structuring the text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=750,
        temperature=0.7
    )

    content = response.choices[0].message.content
    return content

def extract_tasks_to_json(content):
    tasks = []
    task_blocks = re.split(r"(?=Task \d+:)", content)
    for task_block in task_blocks:
            match = re.match(
                r"Task\s+(?P<task_id>\d+):\s+(?P<task_description>.+?)\s*-\s*(?P<start_time>\d{1,2}:\d{2})\s*-\s*(?P<end_time>\d{1,2}:\d{2})\s*\n\s*The needed number of people:\s*(?P<number_of_people>\d+)\s*\n\s*Relevant Skills:\s*(?P<relevant_skills>.+?)\s*\n\s*Material Resources:\s*(?P<material_resources>(?:.+?\(\d+(?: [^\)]+)?\)(?:, )?)*|)",
                task_block
            )
            if match:
                task_id = int(match.group(1))
                task_description = match.group(2).strip()
                start_time = match.group(3).strip()
                end_time = match.group(4).strip()
                number_of_people = int(match.group(5))
                relevant_skills = [skill.strip() for skill in match.group(6).split(",")]
                try:
                  material_resources = [
                      {
                          "resource": resource.split("(")[0].strip(),
                          "quantity": int((resource.split("(")[1].strip(")")).split(" ")[0])
                      } for resource in match.group(7).split(",")
                  ]
                except IndexError:
                  material_resources = []

                tasks.append({
                    "id_of_the_task": task_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "task_description": task_description,
                    "number_of_people": number_of_people,
                    "relevant_skills": relevant_skills,
                    "material_resources": material_resources
                })
    return tasks

def generate_subtasks(task_description, history=None):
    prompt = f"""
   You are a task planner assistant. Divide the following large task into smaller, sequential tasks that can be accomplished in the time bounds given for the global task. Do not consider tasks like breaks.

   If the user says 'hello' or asks something basic you can ask him and say 'What can I help you with?'.

   If the large task can not be accomplished in the given time bounds, please, write that the task requires more time and ask the user to rethink time-bounds, do not plan anything in that case.
   If the task is large and takes a lot of time, refuse to plan it, because it can't be done in one day.
   If the user does not provide time bounds, estimate a time for these task subtasks and set it on your opinion.
   Take into consideration that a limit of a standard workday of a user and distribute tasks accordingly in these boundaries.
   One given task should be done in one day!!! If a given task is huge and takes a lot of time (not one day), refuse to plan it.
   Say that it takes a lot of time to accomplish a task. Estimate how many hours it is gonna take completing this given task, and write in that bounds small subtasks.
   You need to define specific time (like 10:00-11:30)

   When splitting the global task into subtasks, you must take into account that sometimes it is better to do parallel tasking to optimize the routine.
   For example, to buy gloves and to create a cleaning plan can be done in the same time bounds, but simply with different people, so the output should contain 2 subtasks with the same timebounds(buy gloves - 11:00-13:00, create a cleaning plan - 11:00-13:00)
   So, do not always consider consecutive planning, think how the tasks can be done in parallel.
   Think about how much time each subtask requires and write the time bounds for each subtask.

   So what are the things you should do:
   1. If the user congratulates you, you can answer.
   2. Check if given task from user is not big and can be accomplished in one day (for instance, build a house, build a factory, open a business, grow a baby, build a career, read a full library of books, produce a latex at home, mine a mine and find 40 kg gold, I have 1 sewing maching and I need to sew 10000 costumes etc. cannot be done in one day)
   3. Split task into subtasks
   4. Return to user

   Ensure the tasks are logically ordered and not too big or too small. The tasks should collectively accomplish the original task:

   Task: {task_description}

   After dividing the global task into smaller ones in the given time bounds, please, think about how many people should do this task, because sometimes the same subtask can be done by few people and it will be easier and quicker to execute the task.
   But be rational when writing a quantity of people doing the subtask, because sometimes the more people you add, the less effective the work is, so take it logically.


   Also for each subtask think what skills are relevant for perfectly accomplishing every task and write top-3 relevant skills for each subtask.
   Please, note that skills must be relevant for people that likely will do this subtask(for example, if the subtask is likely to be done by Project Manager, then the skills will include Leadership, Critical Thinking etc.)


   You can not assign more than 3 people!
   Include material resources required for each subtask along with the relevant skills. Choose the resources wisely (for example, gloves, car and so on). Also include the number of each resource.
   Do NOT include own personal items (for example, phone, dishes, email account, internet connection etc.). Do NOT include a resource that can be substituted with a free alternative (for example, books).
   If we are using the same resource in a few subtasks, then please assign this resource to the first subtask and do not mention it in other subtasks.

   Please use the following format:
   'Task 1: [Brief explanation of the task] - [Time bounds]
   The needed number of people: [number_of_people]
   Relevant Skills: [Skill 1, Skill 2, Skill 3]
   Material Resources: [Resource 1 (Quantity), Resource 2 (Quantity), Resource 3 (Quantity)]'

   Do not add anything more to the structure I defined. (number_of_people should be only number, without any roles defined after).
   The format of the output I defined should be the last thing you print to the user.
   """

    messages = [{"role": "system", "content": "You are a helpful task planner."}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful task planner."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=750,
        temperature=0.7
    )

    content = response.choices[0].message.content

    if len(content.split(":")) <= 2:
        return content, None, None

    structured_content = structure_final_output(content, history)
    tasks = extract_tasks_to_json(structured_content)

    get_relevant_skills(tasks)
    plan = tasks_to_people(tasks)

    if isinstance(plan, str):
        return plan, None, None

    # Create final output
    final_output = "Here are the subtasks with assigned persons:\n"
    for task in plan:
        names = task['people']
        result = ', '.join(f'{first} {last}' for first, last in names)

        final_output += (
            f"Subtask {task['task_id']}: {task['task_description']}\n"
            f"Assigned to: {result}\n"
            f"Time Bounds: {task['start_at']} – {task['end_at']}\n\n"
            f"Skills: {', '.join(task['skills'])}\n\n"
        )

        if task['resources']:
            final_output += "Resources:\n"
            final_output += '\n'.join(f"{key}: {value}" for resource in task['resources'] for key, value in resource.items())
            final_output += '\n\n'

    return plan

def get_relevant_skills(tasks):
    for p in tasks:
        # sug_skills = ', '.join(p['top_3_skills'])
        sug_skills = ', '.join(p['relevant_skills'])
        input_embedding = embedding_model.embed_query(sug_skills)
        results = db_skills.similarity_search_by_vector(input_embedding, k=3)
        rel_skills = [results[i].page_content for i in range(len(results))]

        p['relevant_skills'] = rel_skills

def find_person_schedule(id):
    for person in schedules:
        if person['id'] == id:
            return person

def get_name_from_id(id):
    for person in employee_data:
        if person['id'] == id:
            return person['name'], person['surname'], person['skills']

def find_person_by_id(id):
    for person in employee_data:
        if person['id'] == id:
            return person

def get_position_employee(jobs):
    job_people = {}
    for job in jobs:
        for person in employee_data:
            if job in person.get('skills', []):
                job_people[person['id']] = person['name']
    return job_people

def user_experience(people, description):
  experience_simularity = dict()
  input_embedding = embedding_model.embed_query(description)

  for person_id in people.keys():
    max_similarity = float('-inf')
    experience = find_person_by_id(person_id)['experience']
    for task in experience:
      task_embedding = embedding_model.embed_query(task)
      similarity = cosine_similarity(input_embedding, task_embedding)
      if similarity > max_similarity:
        max_similarity = similarity

    experience_simularity[person_id] = max_similarity

  sorted_experience = dict(sorted(experience_simularity.items(), key=lambda x: x[1], reverse=True))

  return sorted_experience

def find_people_for_task(relevant, task, bounds, people_for_tasks):
    choice = []
    task_start = datetime.strptime(task['start_time'], "%H:%M").time()
    task_end = datetime.strptime(task['end_time'], "%H:%M").time()

    num_people_needed = task['number_of_people']

    if len(relevant) < num_people_needed:
      return "There are no available people for the task"
    for person_id in relevant:
        schedule = find_person_schedule(person_id)
        day_start = datetime.strptime(schedule['start_at'], "%H:%M").time()
        day_end = datetime.strptime(schedule['end_at'], "%H:%M").time()
        if (task_start < day_start) | (day_end < task_end) :
            continue
        availability = False
        if not schedule['busy']:
            availability = True
        for busy_time in schedule['busy']:
            busy_start = datetime.strptime(busy_time[0], "%H:%M").time()
            busy_end = datetime.strptime(busy_time[1], "%H:%M").time()
            if ((busy_end <= task_start) & (busy_end < task_end)) | ((busy_start > task_start) & (busy_start >= task_end)):
                availability = True
                for task in people_for_tasks:
                    task_bounds = list(task.keys())[0]
                    if person_id in task[task_bounds]:
                        start = task_bounds[0]
                        end = task_bounds[1]
                        if ((task_start < start) & (task_end > end)) | ((task_start > start) & (task_end < end)) | ((task_end < start) & (task_end < end)) | ((task_end > start) & (task_end > end)):
                              availability = False
                              break

            else:
                availability = False
                break
        if availability == True:
            choice.append(person_id)
            num_people_needed -= 1
        if num_people_needed == 0:
            break

    if choice != []:
        people_for_tasks.append({(task_start, task_end): choice})
        return choice
    else:
        return "There are no available people for the task"

def find_person_schedule(person_id):
    for person in schedules:
        if person['id'] == person_id:
            return person
    return None

def assign_task(relevant, bounds):
    choice = ''
    task_start = datetime.datetime.strptime(bounds[0], "%H:%M").time()
    task_end = datetime.datetime.strptime(bounds[1], "%H:%M").time()
    for person_id in relevant:
        schedule = find_person_schedule(person_id)
        if not schedule:
            continue
        day_start = datetime.datetime.strptime(schedule['start_at'], "%H:%M").time()
        day_end = datetime.datetime.strptime(schedule['end_at'], "%H:%M").time()
        if (task_start < day_start) or (day_end < task_end):
            continue
        availability = True
        if not schedule.get('busy'):
            schedule['busy'] = []
        for busy_time in schedule['busy']:
            busy_start = datetime.datetime.strptime(busy_time[0], "%H:%M").time()
            busy_end = datetime.datetime.strptime(busy_time[1], "%H:%M").time()
            if not ((busy_end <= task_start) or (busy_start >= task_end)):
                availability = False
                break
        if availability:
            choice = person_id
            schedule['busy'].append(bounds)
            for idx, person in enumerate(schedules):
                if person['id'] == person_id:
                    schedules[idx] = schedule
                    break
            break
    if choice != '':
        return choice
    else:
        return "There are no available people for the task"
    
def assign_person_task(id, task, bounds):
    employee = find_person_by_id(id)
    updated_employee = {
        'id': employee['id'],
        'name': employee['name'],
        'surname': employee['surname'],
        'skills': employee['skills'],
        'experience':[]
    }
    if 'experience' in employee.keys():
        updated_employee['experience'] = employee['experience']
    updated_employee['experience'].append(task)

    employee_data.remove(employee)
    employee_data.append(updated_employee)


    schedule = find_person_schedule(id)
    schedule['busy'].append(bounds)
    new_schedule = {'id': id, 'start_at': schedule['start_at'], 'end_at': schedule['end_at'], 'busy': schedule['busy']}
    schedules.remove(schedule)
    schedules.append(new_schedule)

def get_name_from_id(person_id):
    for person in employee_data:
        if person['id'] == person_id:
            return person['name'], person['surname'], person.get('skills', [])
    return "", "", []

def tasks_to_people(tasks):
    plan = []
    people_for_tasks = []
    for task in tasks:
        rel_people = get_position_employee(task['relevant_skills'])
        rel_people = user_experience(rel_people, task['task_description'])

        people_ids = find_people_for_task(rel_people, task, (task['start_time'], task['end_time']), people_for_tasks)
        if isinstance(people_ids, str):
            return people_ids

    get_relevant_resources(tasks)
    for i in range(len(people_for_tasks)):
        names = []
        for person in list(people_for_tasks[i].values())[0]:
            name, surname, skills = get_name_from_id(person)
            task_description = tasks[i]['task_description']
            assign_person_task(person, task_description,
                              (tasks[i]['start_time'], tasks[i]['end_time']))
            names.append((name, surname))

        resources = []
        for j in range(len(tasks[i]['real_resources'])):
            res = tasks[i]['real_resources'][j]
            if (not res['resource']) & (res['lack'] == 0):
                name = tasks[i]['material_resources'][j]['resource']
                resources.append({name: "You don't have such resource"})
            elif (not res['resource']) & (res['lack'] > 0):
                name = tasks[i]['material_resources'][j]['resource']
                if res['lack'] == 1:
                    resources.append({name: f"You don't have enough resources (lack of {res['lack']} unit)"})
                else:
                    resources.append({name: f"You don't have enough resources (lack of {res['lack']} units)"})
            elif res['resource']:
                name = tasks[i]['material_resources'][j]['resource']
                if res['quantity'] == 1:
                    resources.append({name: f"You should bring {res['quantity']} unit"})
                else:
                    resources.append({name: f"You should bring {res['quantity']} units"})
        task_plan = {
            'task_id': tasks[i]['id_of_the_task'],
            'task_description': tasks[i]['task_description'],
            'start_at': tasks[i]['start_time'],
            'end_at': tasks[i]['end_time'],
            'people': names,
            'skills': skills,
            'resources': resources
        }

        plan.append(task_plan)
    return plan

st.title("🗂️ Task Planner Application")

st.markdown("""
This application helps you plan large tasks by breaking them down into smaller subtasks and assigning them to relevant team members based on their skills and availability.
""")

# Input Section
with st.form(key='task_form'):
    big_task_description = st.text_area(
        "Enter the description of your main task:",
        height=150,
        value="My task is to plan how to clean the forest."
    )
    submitted = st.form_submit_button("Generate Plan")

if submitted:
    with st.spinner("Generating subtasks..."):
        plan = generate_subtasks(big_task_description)
        st.subheader("Generated Subtasks")

        st.subheader("Task Assignment Plan")

        if isinstance(plan, list):
            for task in plan:
                st.markdown(f"**Task {task['task_id']}: {task['task_description']}**")
                st.markdown(f"- **Time Bounds:** {task['start_at']} - {task['end_at']}")
                st.markdown(f"- **Assigned To:** {', '.join([f'{name} {surname}' for name, surname in task['people']])}")
                st.markdown(f"- **Skills:** {', '.join(task['skills'])}")

                # Resources Section
                if task['resources']:
                    st.markdown("**Resources:**")
                    for resource in task['resources']:
                        for key, value in resource.items():
                            st.markdown(f"- {key}: {value}")
                else:
                    st.markdown("**Resources:** None")

                st.markdown("---")

            display_schedules_table_aggrid(employee_data, schedules)
        else:
            st.error("Plan is not in the expected format. Unable to display tasks.")



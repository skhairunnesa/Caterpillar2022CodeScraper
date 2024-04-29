from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re



service = Service(executable_path="chromedriver.exe")
driver = webdriver.Chrome(service=service)

#using driver to access the github login page
driver.get("https://github.com/login")

# Fill in the login credentials and submit the form
username_input = driver.find_element(By.ID, "login_field")
password_input = driver.find_element(By.ID, "password")
submit_button = driver.find_element(By.NAME, "commit")

username_input.send_keys("YourGithubUsername")#give your github username
password_input.send_keys("YourGithubPassword")#give your github password
submit_button.click()


# Wait for the search bar to load on page
button = WebDriverWait(driver, 15).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-target='qbsearch-input.inputButton']"))
)

# Click on the button
button.click()

#select the search bar into a varible
search_input = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "query-builder-test"))
)

# Using the variable Input text into the search field
search_input.send_keys("keras.layers.advanced_activations.PReLU"+ Keys.ENTER)

#repositories will be showed for the search result we need to move to code section
# Wait for the button to be clickable to move from repository to code
button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.ID, ":Rbpmljb:"))
)

# Click on the button
button.click()

link_urls = [] #initializing a list to store links

#we are iterating using for loop to extract the links of the search result code from multiple result pages
#In each iteration we will be accessing all the links and store them in the above created list
for x in range(3):
    print(x)
    # here links object(varible) has all the search result links present on that page
    links = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, "//a[@data-testid='link-to-search-result']"))
    )
    # Extract href attributes from all links of the page and store them one by one
    link_urls.extend([link.get_attribute('href') for link in links])
    
    #will store the next page button on next_page object then will click it to move to next page   
    next_page = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//a[@aria-label='Next Page']")))

    next_page.click()


tot = 0 # this variable will keep a count of number of search results that comply with our contract 
#next we will iterate over each link in the list and vist the link to validate the code with contract
for link in link_urls:
    #accesing the link
    driver.get(link)
    try:
        #extracting the element that contains the code
        textarea = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//textarea[@data-testid='read-only-cursor-text-area']"))
        )

        # Get the text content of the textarea
        #storing the code extracted from element in a string
        code_content = textarea.text
        #here goes the logic for contract validation 
        #this logic is different for each contract.
        match = re.search(r'.*keras\.layers\.advanced_activations\.PReLU.*', code_content)

        if match:
            line_with_prelu = match.group()
            if 'Activation' not in line_with_prelu:
                tot += 1
    except Exception as e:
        print("Exception occurred:", e)



print("total number: " +str(tot))#number of links that validated the contract 
print("total links: "+str(len(link_urls)))#total number of links that we extracted

tot_links = len(link_urls)
per = tot/tot_links
percent_approve = per*100

if (percent_approve>80):
    print("Contract is Valid ")
else:
    print("Contract is Invalid")
print("total link number: "+str(len(link_urls)))
unique_links = set(link_urls)
unique_links_count = len(unique_links)
print("Number of unique links:", unique_links_count)


# # Wait for a while (optional)
time.sleep(120)

# Quit the browser
driver.quit()

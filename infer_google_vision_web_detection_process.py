import copy
from ikomia import core, dataprocess, utils
from google.cloud import vision
import os
import io
import cv2
import json

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferGoogleVisionWebDetectionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.google_application_credentials = ''
        self.output_folder = str(os.path.join(os.path.dirname(os.path.realpath(__file__)), "output"))
        self.include_geo_results = False

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        self.google_application_credentials = str(params["google_application_credentials"])
        self.output_folder = str(params["output_folder"])
        self.include_geo_results = utils.strtobool(params["cuda"])



    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["google_application_credentials"] = str(self.google_application_credentials)
        params["output_folder"] = str(self.output_folder)
        params["include_geo_results"] = str(self.include_geo_results)


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferGoogleVisionWebDetection(dataprocess.CClassificationTask):

    def __init__(self, name, param):
        dataprocess.CClassificationTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_output(dataprocess.DataDictIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferGoogleVisionWebDetectionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.client = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        self.begin_task_run()

        # Get input
        input = self.get_input(0)
        src_image = input.get_image()

        # Set output
        output_dict = self.get_output(4)

        # Get parameters
        param = self.get_param_object()

        if self.client is None:
            if param.google_application_credentials:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = param.google_application_credentials
            self.client = vision.ImageAnnotatorClient()

        # Convert the NumPy array to a byte stream
        src_image = src_image[..., ::-1] # Convert to bgr
        is_success, image_buffer = cv2.imencode(".jpg", src_image)
        byte_stream = io.BytesIO(image_buffer)
        web_detection_params = vision.WebDetectionParams(include_geo_results=param.include_geo_results)
        image_context = vision.ImageContext(web_detection_params=web_detection_params)
        response = self.client.web_detection(image=byte_stream)

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        # Display results in Ikomia application
        best_match_description = f'{response.web_detection.web_entities[0].description}'
        best_match_score = f'{response.web_detection.web_entities[0].score}'
        self.set_whole_image_results([best_match_description], [best_match_score])

        data_dict = {}
        annotations = response.web_detection
        if annotations.pages_with_matching_images:
            pages_data = []

            for page in annotations.pages_with_matching_images:
                page_data = {
                    "Page url": page.url,
                    "Full Matches": [],
                    "Partial Matches": []
                }

                if page.full_matching_images:
                    for image in page.full_matching_images:
                        page_data["Full Matches"].append({"Image url": image.url})

                if page.partial_matching_images:
                    for image in page.partial_matching_images:
                        page_data["Partial Matches"].append({"Image url": image.url})

                pages_data.append(page_data)

            data_dict["Pages with matching images"] = pages_data

        if annotations.web_entities:
            web_entities_data = []

            for entity in annotations.web_entities:
                entity_data = {
                    "Score": entity.score,
                    "Description": entity.description
                }

                web_entities_data.append(entity_data)

            data_dict["Web entities found"] = web_entities_data

        if annotations.visually_similar_images:
            visually_similar_data = []

            for image in annotations.visually_similar_images:
                visually_similar_data.append({"Image url": image.url})

            data_dict["Visually similar images found"] = visually_similar_data

        output_dict.data = data_dict

        output_path = os.path.join(param.output_folder, "web_detection.json")
        output_path_raw = os.path.join(param.output_folder, "web_detection_raw.json")

        if not os.path.exists(param.output_folder):
            os.makedirs(param.output_folder)
        
        output_dict.save(output_path)

        with open(output_path_raw, "w") as json_file:
            json.dump({'web_detection': str(response.web_detection)}, json_file, indent=4)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferGoogleVisionWebDetectionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_google_vision_web_detection"
        self.info.short_description = "Web Detection detects Web references to an image."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.icon_path = "images/cloud.png"
        self.info.path = "Plugins/Python/Other"
        self.info.version = "1.0.0"
        self.info.authors = "Google"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://cloud.google.com/vision/docs/labels"
        # Code source repository
        self.info.repository = "https://github.com/googleapis/python-vision"
        # Keywords used for search
        self.info.keywords = "Web Detection,Cloud,Vision AI"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OTHER"

    def create(self, param=None):
        # Create algorithm object
        return InferGoogleVisionWebDetection(self.info.name, param)

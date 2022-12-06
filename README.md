# Temperature-prediction
This is a repo containing a project code for temperature prediction.
# Deployment of the machine learning model
1. Download or upgrade [git](https://www.atlassian.com/git/tutorials/install-git)
2. Install [docker]( https://docs.docker.com/desktop/install/mac-install/)
3. Install [neptune.ai](https://neptune.ai/). This package will help us to keep
tracking of ML experiments.
    ### Neptune ai get started
* Create a free [account](https://ui.neptune.ai/auth/realms/neptune/protocol/openid-connect/registrations?client_id=neptune-frontend&redirect_uri=https%3A%2F%2Fapp.neptune.ai%2F-%2Fonboarding&state=e18d6183-a384-42cd-8e3d-ae5738ba63b1&response_mode=fragment&response_type=code&scope=openid&nonce=2cc29941-845b-4dd4-83c1-7a6c6cd72830).

* Install Neptune client library.

      pip install neptune-client
* Add logging to your script.

      import neptune.new as neptune
      run = neptune.init_run("Me/MyProject")
      run["parameters"] = {"lr":0.1, "dropout":0.4}
      run["test_accuracy"] = 0.84
3. Once we create a docker image in a local directory, we will be using our storage resources. 
Since one image can have at least 1.3G, it is better to push into a remote storage.
For that, we create a Hub account where we can create remote repository and ush our docker images.
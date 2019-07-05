#!/bin/bash
docker network create -d bridge elsa
docker run --name ejabberd --hostname ejabberd --network elsa --restart=always -d -p 5222:5222 -v $(pwd)/ejabberd.yml:/home/ejabberd/conf/ejabberd.yml ejabberd/ecs
docker run --name conversejs --hostname conversejs --network elsa --restart=always -d -p 8000:80 openscript/conversejs

sleep 5
docker exec -it ejabberd bin/ejabberdctl register admin ejabberd admin123
docker exec -it ejabberd bin/ejabberdctl register demo ejabberd demo123
docker exec -it ejabberd bin/ejabberdctl register elsa ejabberd elsa123
docker exec -it ejabberd bin/ejabberdctl add_rosteritem elsa ejabberd demo ejabberd demo Friends both
docker exec -it ejabberd bin/ejabberdctl add_rosteritem demo ejabberd elsa ejabberd elsa Friends both

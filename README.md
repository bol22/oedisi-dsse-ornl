
# Sinlge container
1. `oedisi build --system system.json` to build the simulation file under build folder
2. `oedisi run --runner build/system_runner.json` to run the simulation


# Multi-Container
1. `oedisi build -m -p 8766`
2. `cd build`
3. `docker compose up`
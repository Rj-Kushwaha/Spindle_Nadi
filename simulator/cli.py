"""Command-line simulator controls for use without the dashboard."""

from simulator import MachineSimulator


def main() -> None:
    simulator = MachineSimulator()
    print("SPINDLE NADI — Feel the Pulse. Predict the Failure.")
    print("Controls: [n] normal  [d] gradual degradation  [f] near failure")
    print("          [w] set workload       [r] reset       [q] quit")

    while True:
        command = input("\nChoose a control (Enter = next sample): ").strip().lower()
        if command == "q":
            print("Simulator stopped.")
            return
        if command == "n":
            simulator.set_normal()
        elif command == "d":
            simulator.start_degradation()
        elif command == "f":
            simulator.set_near_failure()
        elif command == "r":
            simulator.reset()
        elif command == "w":
            try:
                simulator.set_workload(float(input("Workload (0-100%): ")))
            except ValueError as error:
                print(f"Invalid workload: {error}")
                continue
        elif command:
            print("Unknown control. Choose n, d, f, w, r, or q.")
            continue

        print(simulator.step().format())


if __name__ == "__main__":
    main()

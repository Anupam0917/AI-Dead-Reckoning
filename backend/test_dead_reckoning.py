from backend.dead_reckoning import DeadReckoning


def main():

    print("=" * 60)
    print("DEAD RECKONING TEST")
    print("=" * 60)

    # Starting position
    start_latitude = 26.8467
    start_longitude = 80.9462

    dr = DeadReckoning(
        latitude=start_latitude,
        longitude=start_longitude
    )

    print("\nInitial position:")
    print(f"Latitude  : {dr.latitude:.7f}")
    print(f"Longitude : {dr.longitude:.7f}")

    # Simulate movement:
    #
    # 5 m/s North
    # 2 m/s East
    # for 1 second

    vn_mps = 5.0
    ve_mps = 2.0
    dt = 1.0

    result = dr.update(
        vn_mps=vn_mps,
        ve_mps=ve_mps,
        dt=dt
    )

    print("\nMovement:")
    print(f"North velocity : {vn_mps} m/s")
    print(f"East velocity  : {ve_mps} m/s")
    print(f"Time           : {dt} second")

    print("\nUpdated position:")
    print(f"Latitude  : {result['latitude']:.7f}")
    print(f"Longitude : {result['longitude']:.7f}")

    print("\nLocal displacement:")
    print(f"North : {result['north_m']:.3f} m")
    print(f"East  : {result['east_m']:.3f} m")

    print("\n" + "=" * 60)
    print("DEAD RECKONING TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()
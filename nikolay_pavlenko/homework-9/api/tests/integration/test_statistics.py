def test_get_statistics(client):
    response = client.get("api/v1/statistics")

    assert response.json == {"number_of_apartments": 0}

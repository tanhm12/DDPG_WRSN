Data described as belows:
1. Data file name format:
   The data file name format follows rules as below:
   - For data with remaining energy follows the normal distribution:
        <scenario_index>-<distribution_location_index><number_of_sensor>-<remaining_energy_index>.txt
        where in:
                <scenario_index> in {"s1", "s2", "s3"}, distance between 2 certain nodes increased respectively, not used in case of uniform distribution location
                <distribution_location_index> is "n" if location follows the normal distribution, "u" if the figure is uniform distribution
                <number_of_sensor> is the number of sensor
                <remaining_energy_index> in {"high", "medium", "low"}
        e.g: s1-n20-medium.txt, u70-low.txt

   - For data with remaining energy follows the uniform distribution:
        <distribution_location_index><number_of_sensor>-<sensor_location_index>.txt
        where in:
                <distribution_location_index> is "n" if location follows the normal distribution, "u" if the figure is uniform distribution
                <number_of_sensor> is the number of sensor
                <sensor_location_index> in {"far", "near"}, not used in case of uniform distribution location
        e.g: n20-far.txt, u40.txt

2. In data files:
   There are 4 column meaning respectively:
        x_coordinate y_coordinate power_consumption remaining_energy

3. In Fixed data file:
   E_mc
   E_max
   E_min
   WCE_V
   WCE_U
   WCE_P_MOVE

4. Data Directory structure
   /
   /Data/Normal_distribution_energy /High_energy   /normal_distribution_location /*.txt
                                                   /uniform_distribution_location/*.txt
                                    /Low_energy    /normal_distribution_location /*.txt
                                                   /uniform_distribution_location/*.txt
                                    /Medium_energy /normal_distribution_location /*.txt
                                                   /uniform_distribution_location/*.txt
        /Uniform_distribution_energy/normal_distribution_location /*.txt
                                    /uniform_distribution_location/*.txt

   /Image/Normal_distribution_energy /High_energy   /normal_distribution_location /*.png
                                                    /uniform_distribution_location/*.png
                                     /Low_energy    /normal_distribution_location /*.png
                                                    /uniform_distribution_location/*.png
                                     /Medium_energy /normal_distribution_location /*.png
                                                    /uniform_distribution_location/*.png
         /Uniform_distribution_energy/normal_distribution_location /*.png
                                     /uniform_distribution_location/*.png
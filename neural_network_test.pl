#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;
use Storable;

use lib ".";
use NeuralNetwork;
use data_loader;

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# train with only 5% of training data
my $SHORT_TRAINING = 0;

# test with only 10% of test data
my $SHORT_TEST = 0;

# show difference between expected result and actual result during training
my $DEBUG_TRAINING_RESULT_STEP_BY_STEP = 0;
my $DEBUG_TRAINING_COMPARE_RESULT = 0;

# show weights and latest values before testing
my $DEBUG_SHOW_NEURAL_NETWORK_INTERNALS = 1;

# show stuff during test
my $DEBUG_TEST_RESULT_STEP_BY_STEP = 0;
my $DEBUG_TEST_COMPARE_RESULT = 0;
my $DEBUG_TEST_RESULT_BRIEF = 1;

my $AUTO_AVERAGE = 0;
my @l_sum_result;
my $s__nb_tests = 0;

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

sub main()
{
	$| = 1;

	$SIG{INT} = sub { print "\nCaught a sigint\n";exit 1; };

	my $ro_neural_network = undef;

	if (scalar @ARGV == 1) {

		# use a command-line argument to reload from a previously generated neural_network.storage file

		my $neural_network_file = shift @ARGV;

		$ro_neural_network = retrieve($neural_network_file);

	} else {

		print "\n==> Load training data ...\n";
		my $rh_training_data = data_loader::loadDataFromFiles("train-labels.idx1-ubyte", "train-images.idx3-ubyte");

		my $image_size = $rh_training_data->{width} * $rh_training_data->{height};
		$ro_neural_network = NeuralNetwork::new( [ $image_size, 16, 16, 10 ] );

		print "\n==> Train from data ...\n";
		runEpochTraining(
			$ro_neural_network,
			$rh_training_data->{nb},
			$rh_training_data->{rl_images},
			$rh_training_data->{rl_labels},
			$image_size);

		print "\n==> Save neural network to disk ...\n";
		store $ro_neural_network, 'neural_network.storage';
		open FDW, '>neural_network.dump';
		print FDW Dumper($ro_neural_network);
		close FDW;
	}

	$ro_neural_network->printDebug() if $DEBUG_SHOW_NEURAL_NETWORK_INTERNALS;

	print "\n==> Load test data ...\n";
	my $rl_test_data = data_loader::loadDataFromFiles("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");

	# my $raw_data_empty = '';
	# for (my $i = 0; $i < $rl_test_data->{width} * $rl_test_data->{height}; ++$i) { $raw_data_empty .= chr(0) }
	# $ro_neural_network->setSrcImageDataRaw($raw_data_empty);
	# $ro_neural_network->compute();
	# $ro_neural_network->printResultStr();
	# die();

	print "\n==> Test ...\n";
	my $sum_successes = 0;
	my $rl_images = $rl_test_data->{rl_images};
	my $rl_labels = $rl_test_data->{rl_labels};
	my $image_size = $rl_test_data->{width} * $rl_test_data->{height};
	my $nb_tests = $rl_test_data->{nb} * ($SHORT_TEST?0.1:1);
	for (my $i = 0; $i < $nb_tests; $i++) {
		print "  ".sprintf("%.3f", $i*100/$rl_test_data->{nb})." %\n" if ($i % 1000 == 0);
		my $raw_image = $$rl_images[$i];
		if (length($raw_image) == $image_size) {
			$sum_successes += (runTest($ro_neural_network, $raw_image, $$rl_labels[$i]) ? 1 : 0);
		} else {
			warn "".length($raw_image)."!=".$image_size;
		}
	}
	print "\n";
	print "Successes: $sum_successes / $nb_tests\n";

	return 0;
}
exit main();

#-------------------------------------------

sub runEpochTraining($$$$$)
{
	my ($ro_neural_network, $nb, $rl_images, $rl_labels, $image_size) = @_;

	my $NB_EXAMPLES_PER_ITERATION = 100;
	my $nb_iterations = $nb * ($SHORT_TRAINING?0.05:1) / $NB_EXAMPLES_PER_ITERATION;
	for (my $i = 0; $i < $nb_iterations; $i++) {
		#print "  ".sprintf("%.3f", $i*100/$nb)." %\n" if ($i % 1000 == 0);
		my $sum_costs = 0.0;

		# batch iteration
		for (my $j = 0; $j < $NB_EXAMPLES_PER_ITERATION; ++$j) {

			my $k = $i*$NB_EXAMPLES_PER_ITERATION+$j;

			my $raw_image = $$rl_images[$k];
			if (length($raw_image) == $image_size) {

				my $answer = $$rl_labels[$k];

				$ro_neural_network->setSrcImageDataRaw($raw_image);
				$ro_neural_network->compute();

				my @l_expected_result;
				for (my $z = 0; $z < $ro_neural_network->getNbNeuronsLastLayer(); $z++) {
					push @l_expected_result, ($z==$answer ? 1.0 : 0.0);
				}

				if ($DEBUG_TRAINING_COMPARE_RESULT) {
					$ro_neural_network->printComparedResultsStr(\@l_expected_result);
				}

				if ($DEBUG_TRAINING_RESULT_STEP_BY_STEP && $i == 0 && $j == 0) {
					$ro_neural_network->printDebug();
					system("PAUSE");
				}

				$sum_costs += $ro_neural_network->getCost(\@l_expected_result);

				$ro_neural_network->backpropagate(\@l_expected_result);

			} else {
				warn "".length($raw_image)."!=".$image_size;
			}

		}

		my $infomessage = "  Done: $i / $nb_iterations (iteration average cost: ".($sum_costs*1.0/$NB_EXAMPLES_PER_ITERATION).")\n";
		if ($DEBUG_TRAINING_RESULT_STEP_BY_STEP) {
			$ro_neural_network->printDebug();
			print $infomessage;
			#system("PAUSE");
		} else {
			print $infomessage;
		}

		#$ro_neural_network->changeWeights();
	}
}

#-------------------------------------------

sub runTest($$$)
{
	my ($ro_neural_network, $raw_data, $answer) = @_;

	$ro_neural_network->setSrcImageDataRaw($raw_data);
	$ro_neural_network->compute();
	my @l_result = $ro_neural_network->getResult();

	if ($AUTO_AVERAGE) {
		if ($s__nb_tests == 0) {
			@l_sum_result = @l_result;
		} elsif ($s__nb_tests < 100) {
			for (my $i = 0; $i < scalar(@l_result); ++$i) {
				$l_sum_result[$i] += $l_result[$i];
			}
		} else {
			for (my $i = 0; $i < scalar(@l_result); ++$i) {
				$l_result[$i] -= $l_sum_result[$i]/100.0;
			}
		}
		$s__nb_tests++;
	}

	my $biggest_index = undef;
	for (my $i = 0; $i < scalar(@l_result); ++$i) {
		if (!defined($biggest_index) || ($l_result[$i] > $l_result[$biggest_index])) {
			$biggest_index = $i;
		}
	}

	my @l_expected_result;
	for (my $z = 0; $z < $ro_neural_network->getNbNeuronsLastLayer(); $z++) {
		push @l_expected_result, ($z==$answer ? 1.0 : 0.0);
	}

	if ($DEBUG_TEST_COMPARE_RESULT) {
		print "Expecting: $answer - Got: $biggest_index - ";
		print "Cost ".$ro_neural_network->getCost(\@l_expected_result)." - ";
		print "nb_tests: $s__nb_tests - " if ($AUTO_AVERAGE);
		#print "(".join(" ", @l_sum_result).")" if ($AUTO_AVERAGE);
		$ro_neural_network->printResultStr();
	}

	if ($DEBUG_TEST_RESULT_STEP_BY_STEP) {
		$ro_neural_network->printDebug();
		system("PAUSE");
	}

	if ($DEBUG_TEST_RESULT_BRIEF) {
		print "".($biggest_index == $answer ? "X":".");
	}
	if ($biggest_index == $answer) {
		return 1;
	} else {
		return 0;
	}
}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

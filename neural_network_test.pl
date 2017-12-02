#!/usr/bin/perl

# https://www.youtube.com/watch?v=Ilg3gGewQ5U
# https://www.youtube.com/watch?v=q555kfIFUCM
# https://deeplearning4j.org/opendata
# http://yann.lecun.com/exdb/mnist/

use strict;
use warnings;
use Data::Dumper;
use Storable;

use lib ".";
use NeuralNetwork;
use data_loader;

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# do a fastest, less efficient, training
my $SHORT_TRAINING = 0;

# show difference between expected result and actual result during training
my $DEBUG_TRAINING_COMPARE_RESULT = 0;

# show weights and latest values before testing
my $DEBUG_SHOW_NEURAL_NETWORK_INTERNALS = 1;

# show stuff during test
my $DEBUG_TEST_RESULT_STEP_BY_STEP = 0;
my $DEBUG_TEST_RESULT = 0;
my $DEBUG_TEST_RESULT_BRIEF = 1;

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

sub main()
{
	$| = 1;

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
		my $rl_images = $rh_training_data->{rl_images};
		my $rl_labels = $rh_training_data->{rl_labels};
		for (my $i = 0; $i < $rh_training_data->{nb} * ($SHORT_TRAINING?0.05:1); $i++) {
			print "  ".sprintf("%.3f", $i*100/$rh_training_data->{nb})." %\n" if ($i % 1000 == 0);
			my $raw_image = $$rl_images[$i];
			if (length($raw_image) == $image_size) {
				runTraining($ro_neural_network, $raw_image, $$rl_labels[$i]);
			} else {
				warn "".length($raw_image)."!=".$image_size;
			}
		}

		print "\n==> Save neural network to disk ...\n";
		store $ro_neural_network, 'neural_network.storage';
		open FDW, '>neural_network.dump';
		print FDW Dumper($ro_neural_network);
		close FDW;
	}

	$ro_neural_network->printDebug() if $DEBUG_SHOW_NEURAL_NETWORK_INTERNALS;

	print "\n==> Load test data ...\n";
	my $rl_test_data = data_loader::loadDataFromFiles("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");

	print "\n==> Test ...\n";
	my $sum_successes = 0;
	my $rl_images = $rl_test_data->{rl_images};
	my $rl_labels = $rl_test_data->{rl_labels};
	my $image_size = $rl_test_data->{width} * $rl_test_data->{height};
	for (my $i = 0; $i < $rl_test_data->{nb}; $i++) {
		print "  ".sprintf("%.3f", $i*100/$rl_test_data->{nb})." %\n" if ($i % 1000 == 0);
		my $raw_image = $$rl_images[$i];
		if (length($raw_image) == $image_size) {
			$sum_successes += (runTest($ro_neural_network, $raw_image, $$rl_labels[$i]) ? 1 : 0);
		} else {
			warn "".length($raw_image)."!=".$image_size;
		}
	}
	print "\n";
	print "Successes: $sum_successes / $rl_test_data->{nb}\n";

	return 0;
}
exit main();

#-------------------------------------------

sub runTraining($$$)
{
	my ($ro_neural_network, $raw_data, $answer) = @_;

	$ro_neural_network->setSrcImageDataRaw($raw_data);
	$ro_neural_network->compute();

	my @l_expected_result;
	for (my $i = 0; $i < $ro_neural_network->getNbNeuronsLastLayer(); $i++) {
		push @l_expected_result, ($i==$answer ? 1.0 : 0.0);
	}
	if ($DEBUG_TRAINING_COMPARE_RESULT) {
		$ro_neural_network->printComparedResultsStr(\@l_expected_result);
	}

	$ro_neural_network->backpropagate(\@l_expected_result);
}

#-------------------------------------------

sub runTest($$$)
{
	my ($ro_neural_network, $raw_data, $answer) = @_;

	$ro_neural_network->setSrcImageDataRaw($raw_data);
	$ro_neural_network->compute();
#die();
	my @l_result = $ro_neural_network->getResult();
	my $biggest_index = undef;
	foreach (my $i = 0; $i < scalar(@l_result); ++$i) {
		if (!defined($biggest_index) || ($l_result[$i] > $l_result[$biggest_index])) {
			$biggest_index = $i;
		}
	}

	if ($DEBUG_TEST_RESULT_STEP_BY_STEP) {
		$ro_neural_network->printDebug();
		system("PAUSE");
	}
	if ($DEBUG_TEST_RESULT) {
		print "Expecting: $answer - Got: $biggest_index - ";
		$ro_neural_network->printResultStr();
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

#!/usr/bin/perl

use strict;
use warnings;

package data_loader;

# show loaded images as ascii art
my $DEBUG_DISPLAY_LOADED_IMAGES_WITH_LABELS = 0;

#-------------------------------------------------------------------------------
# "public" function
#-------------------------------------------------------------------------------

sub loadDataFromFiles($$)
{
	my ($labels_file, $images_file) = @_;
	my $rl_labels = readLabelsFile($labels_file);
	my $rh_data   = readImagesFile($images_file);
	my $rl_images = $rh_data->{rl_images};

	my $nb = scalar(@$rl_images);
	die "".scalar(@$rl_labels)."!=".$nb unless (scalar(@$rl_labels) == $nb);
	print "  Loaded $nb images of size $rh_data->{width} x $rh_data->{height}, with labels\n";

	if ($DEBUG_DISPLAY_LOADED_IMAGES_WITH_LABELS) {
		for (my $i = 0; $i < $nb; $i++) {
			print "=> ".$$rl_labels[$i]."\n";
			dumpImage($$rl_images[$i], $rh_data->{width}, $rh_data->{height});
		}
	}

	$rh_data->{rl_labels} = $rl_labels;
	$rh_data->{nb} = $nb;
	return $rh_data;
}

#-------------------------------------------------------------------------------
# "private" stuff
#-------------------------------------------------------------------------------

sub readLabelsFile($)
{
	my ($filepath) = @_;
	open (FD, $filepath) || die "$filepath - !";
	my $var = "";
	my $res_read = read FD, $var, 4*2;
	die unless ($res_read == 4*2);
	my ($magic_number, $nb_items) = unpack("N N", $var);
	print "  File header: $magic_number, $nb_items\n";
	my $rl_labels = [];
	while ((read FD, $var, 1) > 0) {
		my $value = int(unpack("C", $var));
		push @$rl_labels, $value;
	}
	close FD;
	return $rl_labels;
}

#-------------------------------------------

sub readImagesFile($)
{
	my ($filepath) = @_;
	open (FD, $filepath) || die "$filepath - $!";
	my $var = "";
	my $res_read = read FD, $var, 4*4;
	die unless ($res_read == 4*4);
	my ($magic_number, $nb_images, $nb_rows, $nb_columns) = unpack("N N N N", $var);
	print "  File header: $magic_number, $nb_images, $nb_rows, $nb_columns\n";
	my $rh_images = { rl_images => [], width => $nb_columns, height => $nb_rows };
	$var = "";
	while ((read FD, $var, $nb_columns*$nb_rows) > 0) {
		push @{$rh_images->{rl_images}}, $var;
	}
	close FD;
	return $rh_images;
}

#-------------------------------------------
# print image as ascii art

sub dumpImage($)
{
	my ($raw_image, $width, $height) = @_;
	die unless (defined $height);
	for (my $i = 0; $i < $height; ++$i) {
		my @l_table1 = split '', ( substr($raw_image, $i*$width, $width) );
		 # sprintf("%03d", ord($_))
		my @l_table2 = map { ord($_)==0?" ":(ord($_)<50?".":(ord($_)<100?":":(ord($_)<200?"x":"X")))} @l_table1;
		print join('', @l_table2)."\n";
	}
	print "==============================\n";
}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
1;

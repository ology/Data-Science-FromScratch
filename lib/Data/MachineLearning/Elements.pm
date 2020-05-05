package Data::MachineLearning::Elements;

# ABSTRACT: Perl implementation of code from the book "Data Science from Scratch"

our $VERSION = '0.0100';

use Moo;
use strictures 2;
use namespace::clean;

with 'Data::MachineLearning::Elements::LinearAlgebra';
with 'Data::MachineLearning::Elements::Statistics';
with 'Data::MachineLearning::Elements::Probability';
with 'Data::MachineLearning::Elements::Inference';
with 'Data::MachineLearning::Elements::GradientDescent';
with 'Data::MachineLearning::Elements::WorkingWithData';
with 'Data::MachineLearning::Elements::MachineLearning';
with 'Data::MachineLearning::Elements::KNearestNeighbors';
with 'Data::MachineLearning::Elements::NaiveBayes';
with 'Data::MachineLearning::Elements::SimpleLinearRegression';
with 'Data::MachineLearning::Elements::MultipleRegression';
with 'Data::MachineLearning::Elements::LogisticRegression';
with 'Data::MachineLearning::Elements::DecisionTrees';
with 'Data::MachineLearning::Elements::NeuralNetworks';
with 'Data::MachineLearning::Elements::DeepLearning';

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ds = Data::MachineLearning::Elements->new;

=head1 DESCRIPTION

C<Data::MachineLearning::Elements> is a perl implementation of the python code from
L<the book|https://www.oreilly.com/library/view/data-science-from/9781492041122/>.

This code is not meant to be either fast or robust.  It is meant to illustrate
data science algorithms.

Please see individual role documentation for available methods and attributes.
The test files show their basic usage.

=cut

1;
__END__

=head1 SEE ALSO

L<https://www.oreilly.com/library/view/data-science-from/9781492041122/>

L<https://github.com/joelgrus/data-science-from-scratch>

F<t/*> - tests

F<eg/*> - examples

L<Data::MachineLearning::Elements::LinearAlgebra>

L<Data::MachineLearning::Elements::Statistics>

L<Data::MachineLearning::Elements::Probability>

L<Data::MachineLearning::Elements::Inference>

L<Data::MachineLearning::Elements::GradientDescent>

L<Data::MachineLearning::Elements::WorkingWithData>

L<Data::MachineLearning::Elements::MachineLearning>

L<Data::MachineLearning::Elements::KNearestNeighbors>

L<Data::MachineLearning::Elements::NaiveBayes>

L<Data::MachineLearning::Elements::SimpleLinearRegression>

L<Data::MachineLearning::Elements::MultipleRegression>

L<Data::MachineLearning::Elements::LogisticRegression>

L<Data::MachineLearning::Elements::DecisionTrees>

L<Data::MachineLearning::Elements::NeuralNetworks>

L<Data::MachineLearning::Elements::DeepLearning>

=cut

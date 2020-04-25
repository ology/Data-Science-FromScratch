package Data::Science::FromScratch;

# ABSTRACT: Perl implementation of code from the book

our $VERSION = '0.0100';

use Moo;
use strictures 2;
use namespace::clean;

with 'Data::Science::FromScratch::LinearAlgebra';
with 'Data::Science::FromScratch::Statistics';
with 'Data::Science::FromScratch::Probability';
with 'Data::Science::FromScratch::Inference';
with 'Data::Science::FromScratch::GradientDescent';
with 'Data::Science::FromScratch::MachineLearning';

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

=head1 DESCRIPTION

C<Data::Science::FromScratch> is a perl implementation of the python code from
L<the book|https://www.oreilly.com/library/view/data-science-from/9781492041122/>.

Please see individual role documentation for available methods.

=cut

1;
__END__

=head1 SEE ALSO

L<https://www.oreilly.com/library/view/data-science-from/9781492041122/>

L<Data::Science::FromScratch::LinearAlgebra>

L<Data::Science::FromScratch::Statistics>

L<Data::Science::FromScratch::Probability>

L<Data::Science::FromScratch::Inference>

L<Data::Science::FromScratch::GradientDescent>

L<Data::Science::FromScratch::MachineLearning>

=cut

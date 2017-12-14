MAIN = 'ml_hw3' # name of the pdf

OUT_PDF = "#{MAIN}.pdf"
OUT_TEX = "#{MAIN}.tex"
MD_FILES = FileList.new('00-metadata.yaml', '??-*.md')

#PANDOC_HEADER = 'pandoc_templates/ieee-pandoc-template/custom.latex'
BIB_LIB = 'bib.bib'
#CSL = 'CSL/chicago-author-date.csl'
TEX_HEADER='00-header.tex'
CITEPROC_PREAMBLE='99-references-preamble.tex'
#PANDOC_TEMPLATE = 'pandoc_templates/ieee-pandoc-template/template.latex'
CONFIG_FILES = FileList[TEX_HEADER, BIB_LIB] #, CSL, PANDOC_HEADER, PANDOC_TEMPLATE]

pandoc_comm = (
  "pandoc" +
  " --template acmart.latex" +
  " --standalone" +
  " --latex-engine lualatex" +
  " -H #{TEX_HEADER}" +
  #" --biblatex" +
  " --filter pandoc-citeproc" +
  " --filter pandoc-citeproc-preamble" +
  " --bibliography '#{BIB_LIB}'" +
  " -M citeproc-preamble=#{CITEPROC_PREAMBLE}" +
  #" --csl '#{CSL}'" +
  #" --template '#{PANDOC_TEMPLATE}'" +
  " -f markdown" +
  " #{MD_FILES.map {|f| "'#{f}'"} .join(' ')}"
)

#"rules to generate the pdf"
file OUT_PDF => (MD_FILES+CONFIG_FILES) do
  puts 'running pandoc...'
  pandoc = "#{pandoc_comm} -o '#{OUT_PDF}'"
  puts pandoc
  system pandoc
  puts "DONE"
end

#"rules to generate the LaTeX file"
file OUT_TEX => (MD_FILES+CONFIG_FILES) do
  puts 'running pandoc...'
  pandoc = "#{pandoc_comm} -o '#{OUT_TEX}'"
  puts pandoc
  system pandoc
  puts "DONE"
end

task :default => [:compile]

desc "Creates pdf file"
task :compile => [OUT_PDF]

desc "Saves underlying LaTeX file"
task :tex => [OUT_TEX]

desc "Runs 'compile' every second, if no change have been made nothing is done"
task :run do
  all_tasks = [OUT_PDF].map {|t| Rake::Task[t]}
  loop do
    begin
      all_tasks.each {|t| t.reenable}
      all_tasks.each {|t| t.invoke}
    rescue => e
      puts "Error: #{e}, sleeping for 15 seconds."
      Kernel.sleep 15
    end
    Kernel.sleep 1
  end
end
